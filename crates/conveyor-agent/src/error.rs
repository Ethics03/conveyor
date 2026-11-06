//! Errors from running an agent.

use std::error::Error as StdError;

use conveyor_core::providers::ProviderError;
use conveyor_storage::Error as StorageError;

/// Why an agent run could not start or complete.
///
/// Failures after a run starts are also persisted on the run before they are
/// returned. Cancellation and provider responses that end a run early are
/// outcomes, not errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The agent names tools that are not available.
    #[error("Unknown tools: {}", .0.join(", "))]
    UnknownTools(Vec<String>),
    #[error("Cannot run an agent without session messages")]
    EmptyTranscript,
    #[error("Run does not belong to the provided session")]
    RunSessionMismatch,
    /// A stored tool message cannot be sent without its call ID.
    #[error("Tool result message {0} has no tool call ID")]
    MissingToolCallId(String),
    #[error("Provider request failed")]
    Provider {
        run_id: String,
        #[source]
        source: ProviderError,
    },
    #[error(transparent)]
    Storage(#[from] StorageError),
}

/// A result from running an agent.
pub type Result<T> = std::result::Result<T, Error>;

/// An error and its sources joined into one message, as persisted on a run.
pub(crate) fn error_chain(error: &dyn StdError) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}
