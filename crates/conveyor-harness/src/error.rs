//! Errors from the runtime's session and turn APIs.

use conveyor_agent::Error as AgentError;
use conveyor_core::Error as DomainError;
use conveyor_storage::Error as StorageError;

/// Why a runtime operation failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Runtime is closed")]
    Closed,
    #[error("Message content cannot be empty")]
    EmptyContent,
    #[error("Session does not exist: {0}")]
    SessionNotFound(String),
    #[error("Session is not active: {0}")]
    SessionNotActive(String),
    #[error("Session already has an active turn: {0}")]
    TurnActive(String),
    /// The turn's task stopped before reporting, which means it panicked.
    #[error("Turn stopped before reporting its outcome")]
    TurnAborted,
    #[error(transparent)]
    Domain(#[from] DomainError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Agent(#[from] AgentError),
}

/// A result from a runtime operation.
pub type Result<T> = std::result::Result<T, Error>;
