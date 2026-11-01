//! Shared domain records, and provider contracts for Conveyor.
//!
//! This crate has no dependency on storage, concrete tools, or agent
//! orchestration.

mod cancellation;
mod error;
pub mod models;
pub mod providers;

use std::{future::Future, pin::Pin};

pub use cancellation::{CancellationToken, Cancelled, DEFAULT_CANCELLATION_REASON};
pub use error::{Error, Result};

/// A boxed `Send` future, used at object-safe async boundaries.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
