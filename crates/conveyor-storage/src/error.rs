//! Typed persistence errors and conversions that preserve their sources.

use chrono::ParseError;
use conveyor_core::{Error as DomainError, models::ApprovalStatus};
use serde_json::Error as JsonError;
use sqlx::Error as SqlxError;
use tokio::task::JoinError;

/// A recoverable failure from Conveyor's storage API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Store is closed")]
    StoreClosed,
    #[error("SQLite worker stopped before returning the operation result")]
    WorkerStopped,
    #[error("SQLite worker failed")]
    WorkerFailed(#[from] JoinError),
    #[error("SQLite operation failed")]
    Database(#[from] SqlxError),
    #[error("Invalid JSON or serialized enum value")]
    Json(#[from] JsonError),
    #[error("Invalid stored timestamp")]
    Timestamp(#[from] ParseError),
    #[error("Timestamp precision exceeds the storage format's microsecond precision")]
    TimestampPrecision,
    #[error("Database has unsupported schema version {found:?}; expected 1")]
    UnsupportedSchema { found: String },
    #[error("Existing database has no Conveyor schema version")]
    UnversionedDatabase,
    #[error(transparent)]
    Domain(#[from] DomainError),
    #[error("New approvals must be pending and have no resolution timestamp")]
    InvalidNewApproval,
    #[error("Unknown approval: {0}")]
    UnknownApproval(String),
    #[error("Approval {id} is already resolved as {status:?}")]
    ApprovalConflict { id: String, status: ApprovalStatus },
    #[error("Resumed runs must have running status")]
    InvalidResumeStatus,
    #[error("Run {0} is not blocked")]
    RunNotBlocked(String),
}

/// A result from a storage operation.
pub type Result<T> = std::result::Result<T, Error>;
