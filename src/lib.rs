//! Conveyor: a self-hosted runtime for persistent agents.
//!
//! This crate establishes the module boundaries for the runtime.

mod agent;
mod approvals;
mod cancellation;
mod context;
mod error;
pub mod models;
mod providers;
mod runtime;
mod storage;
mod tools;

pub use error::{Error, Result};
