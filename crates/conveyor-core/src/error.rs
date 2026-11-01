//! Domain validation errors, independent of persistence and transport.

/// Invalid domain input supplied to Conveyor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Session title cannot be empty")]
    EmptyTitle,
    #[error("Session title cannot exceed 100 characters")]
    TitleTooLong,
    #[error("Context window must be positive")]
    EmptyContextWindow,
    #[error("Maximum output tokens must be positive")]
    EmptyOutputLimit,
    #[error("Maximum output tokens must be smaller than the context window")]
    OutputExceedsContextWindow,
    #[error("Compaction trigger must be at least {minimum} tokens")]
    CompactionTriggerTooLow { minimum: u32 },
    #[error("API key contains characters not allowed in an HTTP header")]
    InvalidApiKey,
    #[error("Provider base URL must be an absolute http(s) URL")]
    InvalidBaseUrl,
}

/// A result from domain construction or validation.
pub type Result<T> = std::result::Result<T, Error>;
