//! Serialized SQLite operations, transactions, schema checks, and connection lifecycle.

use std::{future::Future, path::Path, pin::Pin, sync::Mutex, time::Duration};

use sqlx::{Connection, SqliteConnection, sqlite::SqliteConnectOptions};
use tokio::{
    sync::{Mutex as AsyncMutex, mpsc, oneshot},
    task::JoinHandle,
};

use super::{queries, rows};
use conveyor_core::models::{self, *};

use crate::{Error, Result};

type QueryFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
type Command = Box<
    dyn for<'a> FnOnce(&'a mut SqliteConnection) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
        + Send,
>;

const SCHEMA: &str = include_str!("schema.sql");
const SCHEMA_VERSION: &str = "1";
const QUEUE_CAPACITY: usize = 64;

#[cfg(test)]
mod tests;

/// Serialized, asynchronous access to a Conveyor SQLite database.
///
/// One owned worker completes each accepted operation, even when its caller's
/// future is dropped. Queue backpressure is cancellation-safe: an operation
/// cancelled before enqueueing does not run. Call [`Self::close`] to stop new
/// requests, drain accepted work, and observe worker/connection shutdown errors.
/// Dropping the store also closes its queue, but cannot await the worker.
pub struct Store {
    sender: Mutex<Option<mpsc::Sender<Command>>>,
    worker: AsyncMutex<Option<JoinHandle<Result<()>>>>,
}

impl Store {
    /// Open or create a database, validating its version before schema changes.
    ///
    /// Use `":memory:"` for an isolated in-memory database. File databases use
    /// WAL, foreign keys, and a five-second busy timeout. Unknown/unversioned
    /// existing schemas are rejected. Connection and schema errors are returned
    /// before a worker is started.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let in_memory = path == Path::new(":memory:");
        let options = SqliteConnectOptions::new()
            .filename(path)
            .in_memory(in_memory)
            .create_if_missing(true)
            .foreign_keys(true)
            .busy_timeout(Duration::from_secs(5));
        let mut connection = SqliteConnection::connect_with(&options).await?;
        if let Err(error) = initialize(&mut connection, in_memory).await {
            connection.close().await?;
            return Err(error);
        }
        let (sender, mut receiver) = mpsc::channel::<Command>(QUEUE_CAPACITY);
        let worker = tokio::spawn(async move {
            while let Some(operation) = receiver.recv().await {
                operation(&mut connection).await;
            }
            connection.close().await?;
            Ok(())
        });
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            worker: AsyncMutex::new(Some(worker)),
        })
    }

    /// Drain submitted operations and close the connection. Repeated calls succeed.
    ///
    /// Dropping this future leaves the worker handle available for a later close.
    /// Operations admitted before closing still complete; later calls fail with
    /// [`Error::StoreClosed`].
    pub async fn close(&self) -> Result<()> {
        let mut worker = self.worker.lock().await;
        drop(self.sender.lock().map_err(|_| Error::WorkerStopped)?.take());
        if let Some(task) = worker.as_mut() {
            let result = task.await;
            *worker = None;
            result??;
        }
        Ok(())
    }

    async fn call<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: for<'a> FnOnce(&'a mut SqliteConnection) -> QueryFuture<'a, T> + Send + 'static,
    {
        let sender = self
            .sender
            .lock()
            .map_err(|_| Error::WorkerStopped)?
            .as_ref()
            .cloned()
            .ok_or(Error::StoreClosed)?;
        let (result_tx, result_rx) = oneshot::channel();
        let command: Command = Box::new(move |connection| {
            Box::pin(async move {
                let result = operation(connection).await;
                // A dropped caller does not cancel a submitted transaction.
                let _ = result_tx.send(result);
            })
        });
        sender
            .send(command)
            .await
            .map_err(|_| Error::WorkerStopped)?;
        drop(sender);
        result_rx.await.map_err(|_| Error::WorkerStopped)?
    }

    async fn write<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: for<'a> FnOnce(&'a mut SqliteConnection) -> QueryFuture<'a, T> + Send + 'static,
    {
        // Single-statement writes use SQLite's implicit transactions through
        // `call`. This wrapper groups operations that must commit together.
        self.call(move |connection| {
            Box::pin(async move {
                let mut transaction = connection.begin_with("BEGIN IMMEDIATE").await?;
                match operation(&mut transaction).await {
                    Ok(value) => {
                        transaction.commit().await?;
                        Ok(value)
                    }
                    Err(error) => {
                        transaction.rollback().await?;
                        Err(error)
                    }
                }
            })
        })
        .await
    }

    /// Insert a session or update its status/timestamp, preserving its saved title.
    pub async fn save_session(&self, session: &Session) -> Result<()> {
        let session = session.clone();
        self.call(move |connection| {
            Box::pin(async move {
                sqlx::query(
                    "INSERT INTO sessions (id, title, title_source, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
                status = excluded.status, updated_at = excluded.updated_at",
                )
                .bind(&session.id)
                .bind(&session.title)
                .bind(session.title_source.as_str())
                .bind(session.status.as_str())
                .bind(rows::timestamp(session.created_at)?)
                .bind(rows::timestamp(session.updated_at)?)
                .execute(connection)
                .await?;
                Ok(())
            })
        })
        .await
    }

    pub async fn get_session(&self, id: &str) -> Result<Option<Session>> {
        let id = id.to_owned();
        self.call(move |connection| {
            Box::pin(async move {
                sqlx::query("SELECT * FROM sessions WHERE id = ?")
                    .bind(id)
                    .fetch_optional(connection)
                    .await?
                    .as_ref()
                    .map(rows::session)
                    .transpose()
            })
        })
        .await
    }

    pub async fn list_sessions(&self, status: Option<SessionStatus>) -> Result<Vec<Session>> {
        self.call(move |connection| {
            Box::pin(async move {
                queries::sessions(status)
                    .fetch_all(connection)
                    .await?
                    .iter()
                    .map(rows::session)
                    .collect()
            })
        })
        .await
    }

    /// Save a validated user title, which subsequent automatic titles cannot replace.
    pub async fn set_session_title(&self, id: &str, title: &str) -> Result<bool> {
        self.set_title(id, title, SessionTitleSource::User).await
    }

    /// Set an automatic title only while the stored title source is `default`.
    pub async fn set_auto_title(&self, id: &str, title: &str) -> Result<bool> {
        self.set_title(id, title, SessionTitleSource::Auto).await
    }

    async fn set_title(&self, id: &str, title: &str, source: SessionTitleSource) -> Result<bool> {
        let (id, title) = (id.to_owned(), title.to_owned());
        self.call(move |connection| {
            Box::pin(async move {
                let title = models::clean_title(&title)?;
                Ok(sqlx::query(
                    "UPDATE sessions SET title = ?, title_source = ?, updated_at = ?
                WHERE id = ? AND (? = 'user' OR title_source = 'default')",
                )
                .bind(title)
                .bind(source.as_str())
                .bind(rows::timestamp(models::now())?)
                .bind(id)
                .bind(source.as_str())
                .execute(connection)
                .await?
                .rows_affected()
                    > 0)
            })
        })
        .await
    }

    pub async fn save_run(&self, run: &Run) -> Result<()> {
        self.save_run_with_events(run, &[]).await
    }

    /// Atomically upsert a run and append its associated lifecycle events.
    pub async fn save_run_with_events(&self, run: &Run, events: &[Event]) -> Result<()> {
        let (run, events) = (run.clone(), events.to_vec());
        self.write(move |connection| {
            Box::pin(async move {
                save_run(connection, &run).await?;
                for event in &events {
                    insert_event(connection, event).await?;
                }
                Ok(())
            })
        })
        .await
    }

    pub async fn get_run(&self, id: &str) -> Result<Option<Run>> {
        let id = id.to_owned();
        self.call(move |connection| {
            Box::pin(async move {
                sqlx::query("SELECT * FROM runs WHERE id = ?")
                    .bind(id)
                    .fetch_optional(connection)
                    .await?
                    .as_ref()
                    .map(rows::run)
                    .transpose()
            })
        })
        .await
    }

    pub async fn list_runs(&self, session_id: &str) -> Result<Vec<Run>> {
        let session_id = session_id.to_owned();
        self.call(move |connection| {
            Box::pin(async move {
                sqlx::query("SELECT * FROM runs WHERE session_id = ? ORDER BY created_at, id")
                    .bind(session_id)
                    .fetch_all(connection)
                    .await?
                    .iter()
                    .map(rows::run)
                    .collect()
            })
        })
        .await
    }

    pub async fn save_message(&self, message: &Message) -> Result<()> {
        self.save_message_with_events(message, &[]).await
    }

    /// Atomically insert a transcript message and append associated events.
    pub async fn save_message_with_events(
        &self,
        message: &Message,
        events: &[Event],
    ) -> Result<()> {
        let (message, events) = (message.clone(), events.to_vec());
        self.write(move |connection| Box::pin(async move {
            sqlx::query("INSERT INTO messages
                (id, session_id, role, content, run_id, name, tool_calls, tool_call_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(&message.id).bind(&message.session_id).bind(message.role.as_str())
                .bind(&message.content).bind(&message.run_id).bind(&message.name)
                .bind(serde_json::to_string(&message.tool_calls)?).bind(&message.tool_call_id)
                .bind(serde_json::to_string(&message.metadata)?).bind(rows::timestamp(message.created_at)?)
                .execute(&mut *connection).await?;
            for event in &events { insert_event(connection, event).await?; }
            Ok(())
        })).await
    }

    pub async fn list_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let session_id = session_id.to_owned();
        self.call(move |connection| {
            Box::pin(async move {
                sqlx::query("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, id")
                    .bind(session_id)
                    .fetch_all(connection)
                    .await?
                    .iter()
                    .map(rows::message)
                    .collect()
            })
        })
        .await
    }

    /// Append an event. There is no update or deletion API for event records.
    pub async fn append_event(&self, event: &Event) -> Result<()> {
        let event = event.clone();
        self.call(move |connection| Box::pin(async move { insert_event(connection, &event).await }))
            .await
    }

    /// List events in timestamp/ID order, applying both filters when provided.
    pub async fn list_events(
        &self,
        session_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<Vec<Event>> {
        let (session_id, run_id) = (session_id.map(str::to_owned), run_id.map(str::to_owned));
        self.call(move |connection| {
            Box::pin(async move {
                queries::events(session_id.as_deref(), run_id.as_deref())
                    .fetch_all(connection)
                    .await?
                    .iter()
                    .map(rows::event)
                    .collect()
            })
        })
        .await
    }

    pub async fn save_approval(&self, approval: &ApprovalRequest) -> Result<()> {
        let approval = approval.clone();
        self.call(move |connection| {
            Box::pin(async move { insert_approval(connection, &approval).await })
        })
        .await
    }

    /// Atomically persist pending approvals, run state, and lifecycle events.
    pub async fn block_run(
        &self,
        run: &Run,
        approvals: &[ApprovalRequest],
        events: &[Event],
    ) -> Result<()> {
        let (run, approvals, events) = (run.clone(), approvals.to_vec(), events.to_vec());
        self.write(move |connection| {
            Box::pin(async move {
                for approval in &approvals {
                    insert_approval(connection, approval).await?;
                }
                save_run(connection, &run).await?;
                for event in &events {
                    insert_event(connection, event).await?;
                }
                Ok(())
            })
        })
        .await
    }

    /// Resume an existing blocked run and append the event in one transaction.
    pub async fn resume_run(&self, run: &Run, event: &Event) -> Result<()> {
        let (run, event) = (run.clone(), event.clone());
        self.write(move |connection| {
            Box::pin(async move {
                if run.status != RunStatus::Running {
                    return Err(Error::InvalidResumeStatus);
                }
                let count = sqlx::query(
                    "UPDATE runs SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status = 'blocked'",
                )
                .bind(run.status.as_str())
                .bind(&run.error)
                .bind(rows::timestamp(run.updated_at)?)
                .bind(&run.id)
                .execute(&mut *connection)
                .await?
                .rows_affected();
                if count != 1 {
                    return Err(Error::RunNotBlocked(run.id));
                }
                insert_event(connection, &event).await
            })
        })
        .await
    }

    pub async fn get_approval(&self, id: &str) -> Result<Option<ApprovalRequest>> {
        let id = id.to_owned();
        self.call(move |connection| Box::pin(async move { get_approval(connection, &id).await }))
            .await
    }

    pub async fn list_approvals(
        &self,
        run_id: Option<&str>,
        status: Option<ApprovalStatus>,
    ) -> Result<Vec<ApprovalRequest>> {
        let run_id = run_id.map(str::to_owned);
        self.call(move |connection| {
            Box::pin(async move {
                queries::approvals(run_id.as_deref(), status)
                    .fetch_all(connection)
                    .await?
                    .iter()
                    .map(rows::approval)
                    .collect()
            })
        })
        .await
    }

    /// Resolve an approval exactly once and append `approval.resolved` atomically.
    ///
    /// Repeating the same decision returns the existing record without a second
    /// event. An opposite decision returns [`Error::ApprovalConflict`].
    pub async fn resolve_approval(
        &self,
        id: &str,
        decision: ApprovalDecision,
    ) -> Result<ApprovalRequest> {
        let id = id.to_owned();
        self.write(move |connection| {
            Box::pin(async move {
                let row = sqlx::query(
                    "UPDATE approvals SET status = ?, resolved_at = ?
                WHERE id = ? AND status = 'pending' RETURNING *",
                )
                .bind(decision.as_str())
                .bind(rows::timestamp(models::now())?)
                .bind(&id)
                .fetch_optional(&mut *connection)
                .await?;
                if let Some(row) = row {
                    let approval = rows::approval(&row)?;
                    let mut event = Event::new(EventType::ApprovalResolved);
                    event.session_id = Some(approval.session_id.clone());
                    event.run_id = Some(approval.run_id.clone());
                    event.payload = serde_json::from_value(serde_json::json!({
                        "approval_id": approval.id, "decision": decision,
                        "tool_call_id": approval.tool_call.id,
                    }))?;
                    insert_event(connection, &event).await?;
                    return Ok(approval);
                }
                let approval = get_approval(connection, &id)
                    .await?
                    .ok_or_else(|| Error::UnknownApproval(id.clone()))?;
                if approval.status == ApprovalStatus::from(decision) {
                    return Ok(approval);
                }
                Err(Error::ApprovalConflict {
                    id,
                    status: approval.status,
                })
            })
        })
        .await
    }
}

async fn validate_version(connection: &mut SqliteConnection) -> Result<()> {
    let tables: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
    )
    .fetch_one(&mut *connection)
    .await?;
    if tables == 0 {
        return Ok(());
    }
    let has_meta: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
    )
    .fetch_one(&mut *connection)
    .await?;
    if !has_meta {
        return Err(Error::UnversionedDatabase);
    }
    let version: Option<String> =
        sqlx::query_scalar("SELECT value FROM meta WHERE key = 'schema_version'")
            .fetch_optional(connection)
            .await?;
    match version {
        Some(found) if found == SCHEMA_VERSION => Ok(()),
        Some(found) => Err(Error::UnsupportedSchema { found }),
        None => Err(Error::UnversionedDatabase),
    }
}

async fn initialize(connection: &mut SqliteConnection, in_memory: bool) -> Result<()> {
    validate_version(connection).await?;
    if !in_memory {
        sqlx::query("PRAGMA journal_mode=WAL")
            .execute(&mut *connection)
            .await?;
    }
    let mut transaction = connection.begin_with("BEGIN IMMEDIATE").await?;
    validate_version(&mut transaction).await?;
    sqlx::raw_sql(SCHEMA).execute(&mut *transaction).await?;
    sqlx::query("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)")
        .bind(SCHEMA_VERSION)
        .execute(&mut *transaction)
        .await?;
    transaction.commit().await?;
    Ok(())
}

async fn save_run(connection: &mut SqliteConnection, run: &Run) -> Result<()> {
    sqlx::query("INSERT INTO runs (id, session_id, agent_id, parent_run_id, status, error, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
        status = excluded.status, error = excluded.error, updated_at = excluded.updated_at")
        .bind(&run.id).bind(&run.session_id).bind(&run.agent_id).bind(&run.parent_run_id)
        .bind(run.status.as_str()).bind(&run.error).bind(rows::timestamp(run.created_at)?)
        .bind(rows::timestamp(run.updated_at)?).execute(connection).await?;
    Ok(())
}

async fn insert_event(connection: &mut SqliteConnection, event: &Event) -> Result<()> {
    sqlx::query(
        "INSERT INTO events (id, type, session_id, run_id, message_id, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&event.id)
    .bind(event.kind.as_str())
    .bind(&event.session_id)
    .bind(&event.run_id)
    .bind(&event.message_id)
    .bind(serde_json::to_string(&event.payload)?)
    .bind(rows::timestamp(event.created_at)?)
    .execute(connection)
    .await?;
    Ok(())
}

async fn insert_approval(
    connection: &mut SqliteConnection,
    approval: &ApprovalRequest,
) -> Result<()> {
    if approval.status != ApprovalStatus::Pending || approval.resolved_at.is_some() {
        return Err(Error::InvalidNewApproval);
    }
    sqlx::query("INSERT INTO approvals (id, session_id, run_id, tool_call, reason, status, created_at, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
        .bind(&approval.id).bind(&approval.session_id).bind(&approval.run_id)
        .bind(serde_json::to_string(&approval.tool_call)?).bind(&approval.reason)
        .bind(approval.status.as_str()).bind(rows::timestamp(approval.created_at)?)
        .bind(approval.resolved_at.map(rows::timestamp).transpose()?).execute(connection).await?;
    Ok(())
}

async fn get_approval(
    connection: &mut SqliteConnection,
    id: &str,
) -> Result<Option<ApprovalRequest>> {
    sqlx::query("SELECT * FROM approvals WHERE id = ?")
        .bind(id)
        .fetch_optional(connection)
        .await?
        .as_ref()
        .map(rows::approval)
        .transpose()
}
