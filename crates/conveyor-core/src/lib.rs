//! Domain records, cancellation, and provider and tool contracts for Conveyor.
//!
//! The Anthropic and OpenAI adapters require the `anthropic` and `openai`
//! features.

mod cancellation;
mod error;
pub mod models;
pub mod providers;
pub mod tools;

use std::{future::Future, pin::Pin};

pub use cancellation::{CancellationToken, Cancelled, DEFAULT_CANCELLATION_REASON};
pub use error::{Error, Result};

/// A boxed `Send` future for object-safe async traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
