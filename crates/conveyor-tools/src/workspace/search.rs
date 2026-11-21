//! `search_files`: ripgrep searches of workspace file names and contents.

use std::{io, path::Path, time::Duration};

use conveyor_core::{
    models::{JsonObject, ToolPermission},
    tools::{Tool, ToolContext, ToolError},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::{ChildStdout, Command};

use super::{
    WorkspaceError, blocking, clamp,
    read::{map_error, schema},
    resolve,
};
use crate::process::{Delimited, PageRequest, ProcessError, RecordFormat, collect_page};

const DEFAULT_SEARCH_LIMIT: i64 = 50;
const MAX_SEARCH_LIMIT: usize = 500;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_SEARCH_ERROR_BYTES: usize = 10_000;
const MAX_SEARCH_MATCH_CHARS: usize = 4_000;
/// The characters of paths and matched text one content search returns.
const MAX_SEARCH_OUTPUT_CHARS: usize = 100_000;
const MAX_SEARCH_PATH_BYTES: usize = 32_768;
/// The bytes past which ripgrep previews a matching line instead of printing
/// it. ripgrep measures lines in bytes but cuts previews by characters, so
/// this is four bytes per kept character: a line it previews always has more
/// characters than are kept, and shorter lines arrive whole to be cut here.
const RIPGREP_MAX_COLUMNS: usize = MAX_SEARCH_MATCH_CHARS * 4;
/// A path, a line number, and a preview of up to [`RIPGREP_MAX_COLUMNS`]
/// characters of as many as four bytes each.
const MAX_SEARCH_LINE_BYTES: usize = MAX_SEARCH_PATH_BYTES + RIPGREP_MAX_COLUMNS * 4 + 1_000;
/// What ripgrep appends to a long matching line it cuts short.
const PREVIEW_SUFFIX: &str = " [... omitted end of long line]";

fn default_path() -> String {
    ".".into()
}

fn default_limit() -> i64 {
    DEFAULT_SEARCH_LIMIT
}

/// What [`SearchFiles`] looks for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchTarget {
    /// File names matching a glob; a bare name matches anywhere in a name.
    #[default]
    Files,
    /// Lines matching a regular expression.
    Content,
}

/// Arguments for [`SearchFiles`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchFilesArgs {
    pub pattern: String,
    /// The directory or file to search; the workspace root by default.
    #[serde(default = "default_path")]
    pub path: String,
    #[serde(default)]
    pub target: SearchTarget,
    /// Results to skip.
    #[serde(default)]
    pub offset: i64,
    /// Most results to return, up to 500.
    #[serde(default = "default_limit")]
    pub limit: i64,
}

/// Searches workspace file names or contents with ripgrep, which must be
/// installed. Results are sorted by path and paged; files ripgrep ignores,
/// such as those in `.gitignore`, are skipped.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchFiles;

impl Tool for SearchFiles {
    const NAME: &'static str = "search_files";
    type Args = SearchFilesArgs;
    type Output = Value;
    type Error = WorkspaceError;

    fn description(&self) -> String {
        "Search workspace files by name or content using ripgrep.".into()
    }

    fn parameters(&self) -> JsonObject {
        schema(json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "target": {"type": "string", "enum": ["files", "content"]},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["pattern"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: WorkspaceError) -> ToolError {
        map_error(error)
    }

    async fn call(
        &self,
        args: SearchFilesArgs,
        context: &ToolContext,
    ) -> Result<Value, WorkspaceError> {
        let path = args.path.clone();
        let (resolved, is_file) = blocking(context, move |workspace, _| {
            let resolved = resolve(workspace, &path)?;
            let metadata = std::fs::metadata(&resolved.path)
                .map_err(|_| WorkspaceError::NotFound(path.clone()))?;
            Ok((resolved, metadata.is_file()))
        })
        .await?;

        let shown = resolved.display();
        let offset = clamp(args.offset, 0, usize::MAX);
        let limit = clamp(args.limit, 1, MAX_SEARCH_LIMIT);
        let search = async {
            match args.target {
                SearchTarget::Files if is_file => {
                    Ok::<_, WorkspaceError>(single_file(&shown, &args.pattern, offset, limit))
                }
                SearchTarget::Files => {
                    let mut command = ripgrep(&resolved.root);
                    command
                        .args(["--files", "--null", "--sort", "path", "-g"])
                        .arg(file_glob(&args.pattern))
                        .arg("--")
                        .arg(root_argument(&resolved.relative));
                    let page = run(command, offset, limit, Files).await?;
                    Ok(SearchResult::files(
                        &args.pattern,
                        &shown,
                        offset,
                        limit,
                        page,
                    ))
                }
                SearchTarget::Content => {
                    let mut command = ripgrep(&resolved.root);
                    command
                        .args([
                            "--null",
                            "--with-filename",
                            "--no-heading",
                            "--line-number",
                            "--max-columns",
                        ])
                        .arg(RIPGREP_MAX_COLUMNS.to_string())
                        .args([
                            "--max-columns-preview",
                            "--sort",
                            "path",
                            "--color",
                            "never",
                        ])
                        .arg("--")
                        .arg(&args.pattern)
                        .arg(root_argument(&resolved.relative));
                    let page = run(command, offset, limit, Matches).await?;
                    Ok(SearchResult::content(
                        &args.pattern,
                        &shown,
                        offset,
                        limit,
                        page,
                    ))
                }
            }
        };
        let result: SearchResult = tokio::select! {
            biased;
            result = search => result?,
            cancelled = context.cancellation().cancelled() => return Err(cancelled.into()),
        };
        super::json(&result)
    }
}

/// ripgrep run from the workspace root. A user configuration file is ignored
/// so the output keeps the format parsed here, `.gitignore` applies even when
/// the workspace is not a git repository, and files that cannot be read are
/// skipped quietly rather than failing the search.
fn ripgrep(root: &Path) -> Command {
    let mut command = Command::new("rg");
    command
        .args(["--no-config", "--no-require-git", "--no-messages"])
        .current_dir(root);
    command
}

fn root_argument(relative: &Path) -> &Path {
    if relative.as_os_str().is_empty() {
        Path::new(".")
    } else {
        relative
    }
}

async fn run<F: RecordFormat>(
    command: Command,
    offset: usize,
    limit: usize,
    format: F,
) -> Result<crate::process::Page<F::Item>, WorkspaceError> {
    let request = PageRequest {
        offset,
        limit,
        timeout: SEARCH_TIMEOUT,
        max_error_bytes: MAX_SEARCH_ERROR_BYTES,
    };
    let page = collect_page(command, request, format)
        .await
        .map_err(|error| match error {
            ProcessError::Spawn(error) if error.kind() == io::ErrorKind::NotFound => {
                WorkspaceError::RipgrepMissing
            }
            other => WorkspaceError::Ripgrep(other.to_string()),
        })?;
    // ripgrep exits with 1 when nothing matched, and with 2 after any error.
    // With `--no-messages`, an unreadable file is an error that prints
    // nothing, so 2 with no message still returns what was found.
    let finished = page.stopped_early
        || matches!(page.exit_code, Some(0 | 1))
        || (page.exit_code == Some(2) && page.errors.is_empty());
    if !finished {
        return Err(WorkspaceError::Ripgrep(if page.errors.is_empty() {
            "process failed".into()
        } else {
            page.errors
        }));
    }
    Ok(page)
}

/// A file-name search of a single file, matched without ripgrep.
fn single_file(shown: &str, pattern: &str, offset: usize, limit: usize) -> SearchResult {
    let glob = file_glob(pattern);
    let name = shown.rsplit('/').next().unwrap_or(shown);
    let files: Vec<String> = if glob_matches(&glob, shown) || glob_matches(&glob, name) {
        vec![shown.to_owned()]
    } else {
        Vec::new()
    };
    let page: Vec<String> = files.iter().skip(offset).take(limit).cloned().collect();
    let truncated = offset.saturating_add(limit) < files.len();
    SearchResult::Files {
        pattern: pattern.to_owned(),
        path: shown.to_owned(),
        next_offset: truncated.then(|| offset + page.len()),
        files: page,
        offset,
        limit,
        total_count: files.len(),
        total_count_is_exact: true,
        truncated,
    }
}

/// The glob for a file-name search: a pattern with no `/` that does not start
/// with `*` matches anywhere in a name; an empty pattern matches everything.
fn file_glob(pattern: &str) -> String {
    let pattern = pattern.trim();
    if pattern.is_empty() {
        "*".to_owned()
    } else if !pattern.contains('/') && !pattern.starts_with('*') {
        format!("*{pattern}*")
    } else {
        pattern.to_owned()
    }
}

/// Shell-style matching: `*` matches any run of characters, `/` included, `?`
/// one character, and `[...]` a set or range, negated by a leading `!`.
fn glob_matches(glob: &str, text: &str) -> bool {
    let glob: Vec<char> = glob.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut g, mut t) = (0, 0);
    let mut backtrack = None;
    while t < text.len() {
        let step = match glob.get(g) {
            Some('*') => {
                backtrack = Some((g, t));
                g += 1;
                continue;
            }
            Some('?') => Some(g + 1),
            Some('[') => match char_class(&glob, g, text[t]) {
                Some((matched, next)) => matched.then_some(next),
                None => (text[t] == '[').then_some(g + 1),
            },
            Some(&literal) => (literal == text[t]).then_some(g + 1),
            None => None,
        };
        match (step, backtrack) {
            (Some(next), _) => {
                g = next;
                t += 1;
            }
            (None, Some((star, start))) => {
                g = star + 1;
                t = start + 1;
                backtrack = Some((star, start + 1));
            }
            (None, None) => return false,
        }
    }
    glob[g..].iter().all(|&character| character == '*')
}

/// Whether the class opening at `glob[open]` matches `character`, and the
/// index after it; `None` when the `[` is not closed and is a literal.
fn char_class(glob: &[char], open: usize, character: char) -> Option<(bool, usize)> {
    let mut index = open + 1;
    let negated = glob.get(index) == Some(&'!');
    if negated {
        index += 1;
    }
    let mut matched = false;
    let mut first = true;
    loop {
        let &current = glob.get(index)?;
        if current == ']' && !first {
            return Some((matched != negated, index + 1));
        }
        first = false;
        if glob.get(index + 1) == Some(&'-') && glob.get(index + 2).is_some_and(|&end| end != ']') {
            let end = glob[index + 2];
            matched |= (current..=end).contains(&character);
            index += 3;
        } else {
            matched |= current == character;
            index += 1;
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "target", rename_all = "lowercase")]
enum SearchResult {
    Files {
        pattern: String,
        path: String,
        files: Vec<String>,
        offset: usize,
        limit: usize,
        total_count: usize,
        total_count_is_exact: bool,
        truncated: bool,
        /// Where the next page starts, when there is one.
        #[serde(skip_serializing_if = "Option::is_none")]
        next_offset: Option<usize>,
    },
    Content {
        pattern: String,
        path: String,
        matches: Vec<Match>,
        offset: usize,
        limit: usize,
        total_count: usize,
        total_count_is_exact: bool,
        truncated: bool,
        /// Where the next page starts, when there is one.
        #[serde(skip_serializing_if = "Option::is_none")]
        next_offset: Option<usize>,
    },
}

impl SearchResult {
    fn files(
        pattern: &str,
        path: &str,
        offset: usize,
        limit: usize,
        page: crate::process::Page<String>,
    ) -> Self {
        Self::Files {
            pattern: pattern.to_owned(),
            path: path.to_owned(),
            next_offset: page.stopped_early.then(|| offset + page.items.len()),
            files: page.items,
            offset,
            limit,
            total_count: page.observed,
            total_count_is_exact: !page.stopped_early,
            truncated: page.stopped_early,
        }
    }

    fn content(
        pattern: &str,
        path: &str,
        offset: usize,
        limit: usize,
        page: crate::process::Page<Match>,
    ) -> Self {
        let mut matches = page.items;
        // Keep the page within the output budget, always returning at least
        // one match so paging moves forward.
        let mut used = 0;
        let kept = matches
            .iter()
            .position(|found| {
                used += found.path.chars().count() + found.text.chars().count();
                used > MAX_SEARCH_OUTPUT_CHARS
            })
            .map_or(matches.len(), |over| over.max(1));
        let budget_cut = kept < matches.len();
        matches.truncate(kept);
        let truncated = page.stopped_early || budget_cut;
        Self::Content {
            pattern: pattern.to_owned(),
            path: path.to_owned(),
            next_offset: truncated.then(|| offset + matches.len()),
            matches,
            offset,
            limit,
            total_count: page.observed,
            total_count_is_exact: !page.stopped_early,
            truncated,
        }
    }
}

#[derive(Debug, Serialize)]
struct Match {
    path: String,
    line: u64,
    text: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    text_truncated: bool,
}

/// ripgrep paths without a leading `./`.
fn normalize_path(bytes: &[u8]) -> String {
    let path = String::from_utf8_lossy(bytes);
    match path.strip_prefix("./") {
        Some(stripped) if path != "." => stripped.to_owned(),
        _ => path.into_owned(),
    }
}

/// `rg --files --null` output: NUL-terminated paths.
struct Files;

impl RecordFormat for Files {
    type Item = String;

    async fn next(
        &mut self,
        reader: &mut Delimited<ChildStdout>,
    ) -> Result<Option<String>, String> {
        loop {
            let record = reader
                .read_until(0, MAX_SEARCH_PATH_BYTES)
                .await
                .map_err(|error| error.to_string())?;
            match record {
                Some(path) if path.is_empty() => continue,
                Some(path) => return Ok(Some(normalize_path(&path))),
                None => return Ok(None),
            }
        }
    }
}

/// `rg --null --line-number` output: one line per match, holding the path,
/// a NUL, and `LINE:TEXT`. A line with no NUL is ripgrep's notice that a
/// binary file matched, which is skipped as binary files are in a directory
/// search.
struct Matches;

impl RecordFormat for Matches {
    type Item = Match;

    async fn next(&mut self, reader: &mut Delimited<ChildStdout>) -> Result<Option<Match>, String> {
        loop {
            let Some(line) = reader
                .read_until(b'\n', MAX_SEARCH_LINE_BYTES)
                .await
                .map_err(|error| error.to_string())?
            else {
                return Ok(None);
            };
            let Some(separator) = line.iter().position(|&byte| byte == 0) else {
                continue;
            };
            let record = String::from_utf8_lossy(&line[separator + 1..]);
            let (number, text) = record
                .split_once(':')
                .ok_or("ripgrep emitted an invalid match")?;
            let number = number
                .parse()
                .map_err(|_| "ripgrep emitted an invalid line number")?;
            let text = text.strip_suffix('\r').unwrap_or(text);
            let (text, previewed) = match text.strip_suffix(PREVIEW_SUFFIX) {
                Some(preview) => (preview, true),
                None => (text, false),
            };
            let clipped = text.chars().count() > MAX_SEARCH_MATCH_CHARS;
            return Ok(Some(Match {
                path: normalize_path(&line[..separator]),
                line: number,
                text: text.chars().take(MAX_SEARCH_MATCH_CHARS).collect(),
                text_truncated: previewed || clipped,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{file_glob, glob_matches};

    #[test]
    fn globs_match_like_shell_patterns() {
        assert!(glob_matches("*.py", "src/a.py"));
        assert!(glob_matches("*models.py*", "models.py"));
        assert!(glob_matches("a?c", "abc"));
        assert!(!glob_matches("a?c", "ac"));
        assert!(glob_matches("[ab]x", "bx"));
        assert!(glob_matches("[a-c]x", "cx"));
        assert!(!glob_matches("[!a-c]x", "cx"));
        assert!(glob_matches("[]]", "]"));
        assert!(glob_matches("a[", "a["));
        assert!(glob_matches("*a*b*", "xaybz"));
        assert!(!glob_matches("*.rs", "main.py"));
    }

    #[test]
    fn bare_names_match_anywhere_in_a_name() {
        assert_eq!(file_glob("models"), "*models*");
        assert_eq!(file_glob("*.py"), "*.py");
        assert_eq!(file_glob("src/*.py"), "src/*.py");
        assert_eq!(file_glob("  "), "*");
    }
}
