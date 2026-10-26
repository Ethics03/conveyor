//! Concrete tools, their registry, and workspace/process execution for Conveyor.
//!
//! Scaffold: tool implementations and registrations will arrive with the tools
//! milestone. Shared tool contracts belong in `conveyor-core`; approval policy
//! and provider orchestration belong in `conveyor-agent`.

mod process;
mod registry;
mod shell;
mod workspace;
