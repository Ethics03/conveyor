//! Conversions between SQLite rows and domain records, including JSON and timestamps.

use chrono::SecondsFormat;
use conveyor_core::models::{ApprovalRequest, Event, Message, Run, Session, Timestamp};
use serde::de::{DeserializeOwned, value::StrDeserializer};
use serde_json::Error as JsonError;
use sqlx::{Row, sqlite::SqliteRow};

use crate::{Error, Result};

pub(super) fn timestamp(value: Timestamp) -> Result<String> {
    if !value.timestamp_subsec_nanos().is_multiple_of(1_000) {
        return Err(Error::TimestampPrecision);
    }
    let precision = if value.timestamp_subsec_nanos() == 0 {
        SecondsFormat::Secs
    } else {
        SecondsFormat::Micros
    };
    Ok(value.to_rfc3339_opts(precision, false))
}

fn date(row: &SqliteRow, name: &str) -> Result<Timestamp> {
    Ok(Timestamp::parse_from_rfc3339(row.try_get(name)?)?)
}

fn enumeration<T: DeserializeOwned>(row: &SqliteRow, name: &str) -> Result<T> {
    Ok(T::deserialize(StrDeserializer::<JsonError>::new(
        row.try_get(name)?,
    ))?)
}

fn json<T: DeserializeOwned>(row: &SqliteRow, name: &str) -> Result<T> {
    Ok(serde_json::from_str(row.try_get(name)?)?)
}

pub(super) fn session(row: &SqliteRow) -> Result<Session> {
    Ok(Session {
        id: row.try_get("id")?,
        title: row.try_get("title")?,
        title_source: enumeration(row, "title_source")?,
        status: enumeration(row, "status")?,
        created_at: date(row, "created_at")?,
        updated_at: date(row, "updated_at")?,
    })
}

pub(super) fn run(row: &SqliteRow) -> Result<Run> {
    Ok(Run {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        agent_id: row.try_get("agent_id")?,
        parent_run_id: row.try_get("parent_run_id")?,
        status: enumeration(row, "status")?,
        error: row.try_get("error")?,
        created_at: date(row, "created_at")?,
        updated_at: date(row, "updated_at")?,
    })
}

pub(super) fn message(row: &SqliteRow) -> Result<Message> {
    Ok(Message {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        role: enumeration(row, "role")?,
        content: row.try_get("content")?,
        run_id: row.try_get("run_id")?,
        name: row.try_get("name")?,
        tool_calls: json(row, "tool_calls")?,
        tool_call_id: row.try_get("tool_call_id")?,
        metadata: json(row, "metadata")?,
        created_at: date(row, "created_at")?,
    })
}

pub(super) fn event(row: &SqliteRow) -> Result<Event> {
    Ok(Event {
        id: row.try_get("id")?,
        kind: enumeration(row, "type")?,
        session_id: row.try_get("session_id")?,
        run_id: row.try_get("run_id")?,
        message_id: row.try_get("message_id")?,
        payload: json(row, "payload")?,
        created_at: date(row, "created_at")?,
    })
}

pub(super) fn approval(row: &SqliteRow) -> Result<ApprovalRequest> {
    let resolved: Option<String> = row.try_get("resolved_at")?;
    Ok(ApprovalRequest {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        run_id: row.try_get("run_id")?,
        tool_call: json(row, "tool_call")?,
        reason: row.try_get("reason")?,
        status: enumeration(row, "status")?,
        created_at: date(row, "created_at")?,
        resolved_at: resolved
            .as_deref()
            .map(Timestamp::parse_from_rfc3339)
            .transpose()?,
    })
}
