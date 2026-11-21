//! Concrete tools, their registry, and workspace/process execution for Conveyor.

mod process;
mod registry;
mod shell;
mod workspace;

pub use registry::{DuplicateTool, ToolRegistry, UnknownTools};
pub use workspace::{
    ReadFile, ReadFileArgs, ReadMany, ReadManyArgs, SearchFiles, SearchFilesArgs, SearchTarget,
    WorkspaceError,
};

/// A registry of every built-in tool: `read_file`, `read_many`, and
/// `search_files`. `search_files` needs ripgrep (`rg`) on the `PATH`.
pub fn default_registry() -> ToolRegistry {
    ToolRegistry::new()
        .with(ReadFile)
        .and_then(|registry| registry.with(ReadMany))
        .and_then(|registry| registry.with(SearchFiles))
        .expect("built-in tool names are unique")
}
