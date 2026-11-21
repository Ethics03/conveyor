//! Workspace path containment and file reading, search, and writing tools.

mod read;
mod search;

use std::{
    ffi::OsString,
    fs, io,
    path::{Component, Path, PathBuf},
};

use conveyor_core::{Cancelled, tools::ToolContext};
use serde::Serialize;
use serde_json::Value;

pub use read::{ReadFile, ReadFileArgs, ReadMany, ReadManyArgs};
pub use search::{SearchFiles, SearchFilesArgs, SearchTarget};

/// Symbolic links followed while resolving one path before giving up.
const MAX_SYMLINKS: usize = 40;

/// Why a workspace tool call failed. The message is shown to the model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorkspaceError {
    #[error("Workspace does not exist: {0}")]
    MissingWorkspace(String),
    #[error("Path escapes workspace: {0}")]
    Escapes(String),
    #[error("Path not found: {0}")]
    NotFound(String),
    #[error("Not a file: {0}")]
    NotAFile(String),
    #[error("{0} is not valid UTF-8 text")]
    NotText(String),
    #[error("Offset {offset} exceeds the file's {lines} lines")]
    OffsetPastEnd { offset: usize, lines: usize },
    #[error("Cannot access {path}: {source}")]
    Io { path: String, source: io::Error },
    #[error("ripgrep is required for search_files")]
    RipgrepMissing,
    #[error("ripgrep failed: {0}")]
    Ripgrep(String),
    #[error("Tool output could not be encoded: {0}")]
    Encode(#[from] serde_json::Error),
    #[error("Tool stopped unexpectedly")]
    Stopped,
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}

impl WorkspaceError {
    fn io(path: &str, source: io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            source,
        }
    }
}

/// A requested path resolved inside the workspace.
#[derive(Debug)]
pub(crate) struct Resolved {
    /// The canonical workspace root.
    pub(crate) root: PathBuf,
    /// The path with every symbolic link it reaches followed.
    pub(crate) path: PathBuf,
    /// The path relative to the root; empty for the root itself.
    pub(crate) relative: PathBuf,
}

impl Resolved {
    /// The relative path as shown to the model, with `/` separators and `.`
    /// for the root.
    pub(crate) fn display(&self) -> String {
        let parts: Vec<_> = self
            .relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect();
        if parts.is_empty() {
            ".".to_owned()
        } else {
            parts.join("/")
        }
    }
}

/// Resolve `path`, relative to `workspace` unless absolute, and refuse it when
/// it leads outside the workspace.
///
/// Symbolic links are followed even when their target does not exist, so a
/// link can neither be read through nor created through to escape. Parts of
/// the path that do not exist yet are resolved lexically. The check holds for
/// the file system as it is now; a link created afterwards is not seen.
pub(crate) fn resolve(workspace: &Path, path: &str) -> Result<Resolved, WorkspaceError> {
    let root = fs::canonicalize(workspace)
        .map_err(|_| WorkspaceError::MissingWorkspace(workspace.display().to_string()))?;
    let candidate = root.join(path);
    let resolved = real_path(&candidate).map_err(|source| WorkspaceError::io(path, source))?;
    let relative = resolved
        .strip_prefix(&root)
        .map_err(|_| WorkspaceError::Escapes(path.to_owned()))?
        .to_path_buf();
    Ok(Resolved {
        root,
        path: resolved,
        relative,
    })
}

enum Part {
    Root(PathBuf),
    Current,
    Parent,
    Name(OsString),
}

fn parts(path: &Path) -> Vec<Part> {
    path.components()
        .map(|component| match component {
            Component::Prefix(_) | Component::RootDir => Part::Root(component.as_os_str().into()),
            Component::CurDir => Part::Current,
            Component::ParentDir => Part::Parent,
            Component::Normal(name) => Part::Name(name.to_owned()),
        })
        .collect()
}

/// `path` with `.` and `..` applied and every existing symbolic link followed,
/// including links whose target is missing. Components that do not exist are
/// kept as written.
fn real_path(path: &Path) -> io::Result<PathBuf> {
    let mut pending = parts(path);
    pending.reverse();
    let mut resolved = PathBuf::new();
    let mut links = 0;
    while let Some(part) = pending.pop() {
        match part {
            Part::Root(root) => resolved.push(root),
            Part::Current => {}
            Part::Parent => {
                resolved.pop();
            }
            Part::Name(name) => {
                let next = resolved.join(&name);
                let is_link = fs::symlink_metadata(&next)
                    .map(|metadata| metadata.file_type().is_symlink())
                    .unwrap_or(false);
                if !is_link {
                    resolved = next;
                    continue;
                }
                links += 1;
                if links > MAX_SYMLINKS {
                    return Err(io::Error::other("too many levels of symbolic links"));
                }
                let target = fs::read_link(&next)?;
                // A relative target is resolved from the link's directory,
                // which `resolved` still is; an absolute one starts over.
                let mut target_parts = parts(&target);
                target_parts.reverse();
                pending.extend(target_parts);
            }
        }
    }
    Ok(resolved)
}

/// Run blocking file system work for a tool call off the async workers.
///
/// Cancellation returns at once; the work runs to its end on its thread, so it
/// must be bounded. A result that is ready when cancellation arrives wins.
pub(crate) async fn blocking<T, F>(context: &ToolContext, work: F) -> Result<T, WorkspaceError>
where
    T: Send + 'static,
    F: FnOnce(&Path) -> Result<T, WorkspaceError> + Send + 'static,
{
    let workspace = context.workspace().to_path_buf();
    let task = tokio::task::spawn_blocking(move || work(&workspace));
    tokio::select! {
        biased;
        joined = task => joined.map_err(|_| WorkspaceError::Stopped)?,
        cancelled = context.cancellation().cancelled() => Err(cancelled.into()),
    }
}

/// Encode a tool's structured output.
pub(crate) fn json(output: &impl Serialize) -> Result<Value, WorkspaceError> {
    Ok(serde_json::to_value(output)?)
}

/// Clamp a model-supplied integer into `min..=max`.
pub(crate) fn clamp(value: i64, min: usize, max: usize) -> usize {
    usize::try_from(value).map_or(min, |value| value.clamp(min, max))
}
