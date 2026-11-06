//! Agent orchestration over provider, tool, approval, and storage contracts.

mod approvals;
mod context;
mod error;
mod run;
mod titles;
mod tool_batch;

pub use context::build_provider_messages;
pub use error::{Error, Result};
pub use run::{DEFAULT_MAX_ITERATIONS, RunRequest, run_agent};
