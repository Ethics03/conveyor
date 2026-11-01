//! List queries with direct predicates so SQLite can use available indexes.

use conveyor_core::models::{ApprovalStatus, SessionStatus};
use sqlx::{Sqlite, query::Query, sqlite::SqliteArguments};

type Select<'a> = Query<'a, Sqlite, SqliteArguments>;

pub(super) fn sessions(status: Option<SessionStatus>) -> Select<'static> {
    match status {
        Some(status) => {
            sqlx::query("SELECT * FROM sessions WHERE status = ? ORDER BY created_at, id")
                .bind(status.as_str())
        }
        None => sqlx::query("SELECT * FROM sessions ORDER BY created_at, id"),
    }
}

pub(super) fn events<'a>(session_id: Option<&'a str>, run_id: Option<&'a str>) -> Select<'a> {
    match (session_id, run_id) {
        (Some(session_id), Some(run_id)) => sqlx::query(
            "SELECT * FROM events WHERE session_id = ? AND run_id = ? ORDER BY created_at, id",
        )
        .bind(session_id)
        .bind(run_id),
        (Some(session_id), None) => {
            sqlx::query("SELECT * FROM events WHERE session_id = ? ORDER BY created_at, id")
                .bind(session_id)
        }
        (None, Some(run_id)) => {
            sqlx::query("SELECT * FROM events WHERE run_id = ? ORDER BY created_at, id")
                .bind(run_id)
        }
        (None, None) => sqlx::query("SELECT * FROM events ORDER BY created_at, id"),
    }
}

pub(super) fn approvals(run_id: Option<&str>, status: Option<ApprovalStatus>) -> Select<'_> {
    match (run_id, status) {
        (Some(run_id), Some(status)) => sqlx::query(
            "SELECT * FROM approvals WHERE run_id = ? AND status = ? ORDER BY created_at, id",
        )
        .bind(run_id)
        .bind(status.as_str()),
        (Some(run_id), None) => {
            sqlx::query("SELECT * FROM approvals WHERE run_id = ? ORDER BY created_at, id")
                .bind(run_id)
        }
        (None, Some(status)) => {
            sqlx::query("SELECT * FROM approvals WHERE status = ? ORDER BY created_at, id")
                .bind(status.as_str())
        }
        (None, None) => sqlx::query("SELECT * FROM approvals ORDER BY created_at, id"),
    }
}
