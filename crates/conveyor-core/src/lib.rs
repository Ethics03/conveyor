//! Shared domain records and provider contracts for Conveyor.
//!
//! Domain records and validation are implemented. Provider contracts and adapters
//! will follow. This crate has no dependency on storage or agent orchestration.

mod cancellation;
mod error;
pub mod models;
mod providers;

pub use cancellation::{CancellationToken, Cancelled, DEFAULT_CANCELLATION_REASON};
pub use error::{Error, Result};
