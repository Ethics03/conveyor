//! Domain validation errors, independent of persistence and transport.

/// Invalid domain input supplied to Conveyor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Session title cannot be empty")]
    EmptyTitle,
    #[error("Session title cannot exceed 100 characters")]
    TitleTooLong,
}

/// A result from domain construction or validation.
pub type Result<T> = std::result::Result<T, Error>;
