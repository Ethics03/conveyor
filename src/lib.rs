//! Conveyor: a self-hosted runtime for persistent agents.
//!
//! Domain records and durable SQLite storage form the first migration milestone.
//! The agent runtime and provider/tool adapters will follow incrementally.

mod agent;
mod approvals;
mod cancellation;
mod context;
mod error;
pub mod models;
mod providers;
mod runtime;
pub mod storage;
mod tools;

pub use error::{Error, Result};
pub use storage::Store;
