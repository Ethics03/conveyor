//! Shared resource ownership, session and turn APIs, interruption, and shutdown.

use std::{
    collections::HashMap,
    pin::pin,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use conveyor_agent::{Error as AgentError, RunRequest, run_agent};
use conveyor_core::{
    CancellationToken,
    models::{Agent, Event, EventType, Message, MessageRole, RunOutcome, Session, SessionStatus},
    providers::Provider,
};
use conveyor_storage::Store;
use tokio::sync::{Notify, oneshot};

use crate::{Error, Result};

/// The cancellation reason given to turns still running at shutdown.
pub const RUNTIME_SHUTDOWN_REASON: &str = "Runtime is shutting down";

/// Owns the store and provider, and every turn running on them.
///
/// Each turn runs as its own task, so callers and other sessions never wait
/// on it. A session has at most one active turn. [`close`](Self::close) stops
/// new turns, cancels running ones, waits for them to persist their outcome,
/// and closes the store.
#[derive(Debug)]
pub struct Runtime {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    store: Store,
    provider: Arc<dyn Provider>,
    turns: Mutex<Turns>,
    idle: Notify,
}

#[derive(Debug, Default)]
struct Turns {
    closed: bool,
    active: HashMap<String, CancellationToken>,
}

impl Runtime {
    pub fn new(store: Store, provider: Arc<dyn Provider>) -> Self {
        Self {
            shared: Arc::new(Shared {
                store,
                provider,
                turns: Mutex::default(),
                idle: Notify::new(),
            }),
        }
    }

    pub fn store(&self) -> &Store {
        &self.shared.store
    }

    /// Create and persist a session, with a validated user title if given.
    pub async fn create_session(&self, title: Option<&str>) -> Result<Session> {
        self.ensure_open()?;
        let session = match title {
            Some(title) => Session::with_title(title)?,
            None => Session::default(),
        };
        self.shared.store.save_session(&session).await?;
        Ok(session)
    }

    /// Save `content` as a user message and start a turn for it in the background.
    ///
    /// The returned handle reports the outcome; dropping it leaves the turn
    /// running. Fails without saving anything when the runtime is closed, the
    /// content is blank, the session is missing or archived, or the session
    /// already has an active turn.
    pub async fn start_turn(
        &self,
        agent: Agent,
        session_id: &str,
        content: impl Into<String>,
    ) -> Result<TurnHandle> {
        self.ensure_open()?;
        let content = content.into();
        if content.trim().is_empty() {
            return Err(Error::EmptyContent);
        }
        let session = self
            .shared
            .store
            .get_session(session_id)
            .await?
            .ok_or_else(|| Error::SessionNotFound(session_id.to_owned()))?;
        if session.status != SessionStatus::Active {
            return Err(Error::SessionNotActive(session.id));
        }

        let guard = self.register_turn(&session.id)?;
        let message = Message::new(&session.id, MessageRole::User, content);
        let mut created = Event::new(EventType::MessageCreated);
        created.session_id = Some(session.id.clone());
        created.message_id = Some(message.id.clone());
        created.payload.insert("role".into(), "user".into());
        self.shared
            .store
            .save_message_with_events(&message, &[created])
            .await?;

        let (sender, receiver) = oneshot::channel();
        let shared = Arc::clone(&self.shared);
        let session_id = session.id.clone();
        tokio::spawn(async move {
            let outcome = run_agent(RunRequest::new(
                &agent,
                &session,
                shared.provider.as_ref(),
                &shared.store,
                &guard.cancellation,
            ))
            .await;
            // Release the session before reporting, so the caller can start
            // the next turn as soon as it sees this outcome.
            drop(guard);
            let _ = sender.send(outcome);
        });
        Ok(TurnHandle {
            session_id,
            outcome: receiver,
        })
    }

    /// Start a turn and wait for its outcome.
    pub async fn run_turn(
        &self,
        agent: Agent,
        session_id: &str,
        content: impl Into<String>,
    ) -> Result<RunOutcome> {
        self.start_turn(agent, session_id, content)
            .await?
            .outcome()
            .await
    }

    /// Request cancellation of the session's active turn, if it has one.
    ///
    /// The turn persists its cancelled outcome before its handle resolves.
    pub fn interrupt(&self, session_id: &str, reason: &str) -> Result<bool> {
        let turns = self.shared.lock();
        if turns.closed {
            return Err(Error::Closed);
        }
        Ok(match turns.active.get(session_id) {
            Some(cancellation) => {
                cancellation.cancel(reason);
                true
            }
            None => false,
        })
    }

    /// Whether the session has a turn running.
    pub fn is_active(&self, session_id: &str) -> bool {
        self.shared.lock().active.contains_key(session_id)
    }

    /// Reject new turns, cancel running ones, wait for them, then close the store.
    ///
    /// Repeated calls succeed. If this future is dropped, turns stay cancelled
    /// and a later call finishes the shutdown.
    pub async fn close(&self) -> Result<()> {
        {
            let mut turns = self.shared.lock();
            turns.closed = true;
            for cancellation in turns.active.values() {
                cancellation.cancel(RUNTIME_SHUTDOWN_REASON);
            }
        }
        loop {
            // Register before checking so the last turn's exit cannot be missed.
            let mut idle = pin!(self.shared.idle.notified());
            idle.as_mut().enable();
            if self.shared.lock().active.is_empty() {
                break;
            }
            idle.await;
        }
        self.shared.store.close().await?;
        Ok(())
    }

    fn ensure_open(&self) -> Result<()> {
        match self.shared.lock().closed {
            true => Err(Error::Closed),
            false => Ok(()),
        }
    }

    fn register_turn(&self, session_id: &str) -> Result<ActiveTurn> {
        let mut turns = self.shared.lock();
        if turns.closed {
            return Err(Error::Closed);
        }
        if turns.active.contains_key(session_id) {
            return Err(Error::TurnActive(session_id.to_owned()));
        }
        let cancellation = CancellationToken::new();
        turns
            .active
            .insert(session_id.to_owned(), cancellation.clone());
        Ok(ActiveTurn {
            shared: Arc::clone(&self.shared),
            session_id: session_id.to_owned(),
            cancellation,
        })
    }
}

impl Shared {
    // Critical sections only insert, remove, or read entries, so a poisoned
    // lock still guards consistent data.
    fn lock(&self) -> MutexGuard<'_, Turns> {
        self.turns.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Holds a session's active-turn slot and releases it on every exit path.
struct ActiveTurn {
    shared: Arc<Shared>,
    session_id: String,
    cancellation: CancellationToken,
}

impl Drop for ActiveTurn {
    fn drop(&mut self) {
        let mut turns = self.shared.lock();
        turns.active.remove(&self.session_id);
        if turns.active.is_empty() {
            self.shared.idle.notify_waiters();
        }
    }
}

/// The outcome of a turn running in the background.
#[derive(Debug)]
pub struct TurnHandle {
    session_id: String,
    outcome: oneshot::Receiver<std::result::Result<RunOutcome, AgentError>>,
}

impl TurnHandle {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Wait for the turn to finish and persist its outcome.
    ///
    /// Dropping this future does not cancel the turn.
    pub async fn outcome(self) -> Result<RunOutcome> {
        match self.outcome.await {
            Ok(outcome) => Ok(outcome?),
            Err(_) => Err(Error::TurnAborted),
        }
    }
}
