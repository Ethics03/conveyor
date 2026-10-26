//! Typed errors for domain construction and validation.

/// A recoverable failure from Conveyor's library API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Session title cannot be empty")]
    EmptyTitle,
    #[error("Session title cannot exceed 100 characters")]
    TitleTooLong,
}

/// The result type used by Conveyor's library API.
pub type Result<T> = std::result::Result<T, Error>;
