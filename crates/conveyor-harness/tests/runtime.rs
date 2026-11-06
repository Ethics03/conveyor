use std::{
    collections::VecDeque,
    future,
    sync::{Arc, Mutex},
    time::Duration,
};

use conveyor_core::{
    BoxFuture, Error as DomainError,
    models::{Agent, EventType, MessageRole, RunStatus, Session, SessionStatus},
    providers::{
        ModelLimits, Provider, ProviderError, ProviderRequest, ProviderResponse, fake::FakeProvider,
    },
};
use conveyor_harness::{Error, RUNTIME_SHUTDOWN_REASON, Runtime};
use conveyor_storage::{Error as StorageError, Store};
use tokio::{
    sync::{Barrier, Notify},
    time::timeout,
};

const WAIT: Duration = Duration::from_secs(5);

async fn memory_runtime(provider: Arc<dyn Provider>) -> Runtime {
    Runtime::new(Store::open(":memory:").await.unwrap(), provider)
}

/// Answers each request from a script; `None` waits until the request is dropped.
#[derive(Debug, Default)]
struct GateProvider {
    script: Mutex<VecDeque<Option<&'static str>>>,
    started: Notify,
}

impl GateProvider {
    fn new(script: impl IntoIterator<Item = Option<&'static str>>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().collect()),
            started: Notify::new(),
        })
    }
}

impl Provider for GateProvider {
    fn name(&self) -> &str {
        "gate"
    }

    fn model_limits(&self, _model: Option<&str>) -> ModelLimits {
        ModelLimits::new(200_000, 4_096).unwrap()
    }

    fn generate(
        &self,
        _request: ProviderRequest,
    ) -> BoxFuture<'_, Result<ProviderResponse, ProviderError>> {
        let step = self.script.lock().unwrap().pop_front().flatten();
        Box::pin(async move {
            self.started.notify_one();
            match step {
                Some(content) => Ok(ProviderResponse::message(content)),
                None => future::pending().await,
            }
        })
    }
}

/// Answers only once `parties` requests are in flight at the same time.
#[derive(Debug)]
struct RendezvousProvider {
    barrier: Barrier,
}

impl Provider for RendezvousProvider {
    fn name(&self) -> &str {
        "rendezvous"
    }

    fn model_limits(&self, _model: Option<&str>) -> ModelLimits {
        ModelLimits::new(200_000, 4_096).unwrap()
    }

    fn generate(
        &self,
        _request: ProviderRequest,
    ) -> BoxFuture<'_, Result<ProviderResponse, ProviderError>> {
        Box::pin(async move {
            self.barrier.wait().await;
            Ok(ProviderResponse::message("together"))
        })
    }
}

#[tokio::test]
async fn a_turn_persists_its_input_and_runs_the_agent() {
    let provider = Arc::new(FakeProvider::new(["Hello from Conveyor"]));
    let runtime = memory_runtime(provider.clone()).await;
    let session = runtime.create_session(Some("Runtime test")).await.unwrap();

    let outcome = runtime
        .run_turn(Agent::default(), &session.id, "  Keep this spacing.  ")
        .await
        .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert_eq!(
        outcome.final_message.unwrap().content,
        "Hello from Conveyor"
    );
    let messages = runtime.store().list_messages(&session.id).await.unwrap();
    let roles: Vec<_> = messages.iter().map(|message| message.role).collect();
    assert_eq!(roles, [MessageRole::User, MessageRole::Assistant]);
    assert_eq!(messages[0].content, "  Keep this spacing.  ");
    assert_eq!(messages[0].run_id, None);
    assert_eq!(messages[1].run_id.as_deref(), Some(outcome.run.id.as_str()));

    let events = runtime
        .store()
        .list_events(Some(&session.id), None)
        .await
        .unwrap();
    let kinds: Vec<_> = events.iter().map(|event| event.kind).collect();
    assert_eq!(
        kinds,
        [
            EventType::MessageCreated,
            EventType::RunStarted,
            EventType::MessageCreated,
            EventType::RunFinished
        ]
    );
    assert_eq!(
        events[0].message_id.as_deref(),
        Some(messages[0].id.as_str())
    );
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    let user = requests[0].messages.last().unwrap().content().to_owned();
    assert!(user.starts_with("Message timestamp (UTC): "), "{user}");
    assert!(user.ends_with("\n\n  Keep this spacing.  "), "{user}");
    assert!(!runtime.is_active(&session.id));
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn sessions_persist_across_reopen_and_titles_are_validated() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("conveyor.sqlite");
    let runtime = Runtime::new(
        Store::open(&path).await.unwrap(),
        Arc::new(FakeProvider::default()),
    );
    let titled = runtime
        .create_session(Some("  Fix   timeouts "))
        .await
        .unwrap();
    let untitled = runtime.create_session(None).await.unwrap();
    let error = runtime.create_session(Some("   ")).await.unwrap_err();
    assert!(matches!(error, Error::Domain(DomainError::EmptyTitle)));
    runtime.close().await.unwrap();

    let store = Store::open(&path).await.unwrap();
    assert_eq!(
        store.get_session(&titled.id).await.unwrap(),
        Some(titled.clone())
    );
    assert_eq!(titled.title, "Fix timeouts");
    assert_eq!(
        store.get_session(&untitled.id).await.unwrap(),
        Some(untitled)
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn interrupting_cancels_the_provider_call_and_frees_the_session() {
    let provider = GateProvider::new([None, Some("Continued after cancellation.")]);
    let runtime = memory_runtime(provider.clone()).await;
    let session = runtime.create_session(None).await.unwrap();

    let turn = runtime
        .start_turn(Agent::default(), &session.id, "Wait for the provider.")
        .await
        .unwrap();
    assert_eq!(turn.session_id(), session.id);
    timeout(WAIT, provider.started.notified()).await.unwrap();
    assert!(runtime.is_active(&session.id));
    assert!(runtime.interrupt(&session.id, "Stop provider").unwrap());

    let outcome = timeout(WAIT, turn.outcome()).await.unwrap().unwrap();
    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(outcome.run.error.as_deref(), Some("Stop provider"));
    let kinds: Vec<_> = runtime
        .store()
        .list_events(None, Some(&outcome.run.id))
        .await
        .unwrap()
        .into_iter()
        .map(|event| event.kind)
        .collect();
    assert_eq!(kinds, [EventType::RunStarted, EventType::RunCancelled]);

    assert!(!runtime.interrupt(&session.id, "Nothing running").unwrap());
    let next = runtime
        .run_turn(Agent::default(), &session.id, "Continue.")
        .await
        .unwrap();
    assert_eq!(next.run.status, RunStatus::Finished);
    assert_eq!(
        next.final_message.unwrap().content,
        "Continued after cancellation."
    );
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn a_session_runs_one_turn_at_a_time() {
    let provider = GateProvider::new([None]);
    let runtime = memory_runtime(provider.clone()).await;
    let session = runtime.create_session(None).await.unwrap();
    let _turn = runtime
        .start_turn(Agent::default(), &session.id, "First")
        .await
        .unwrap();
    timeout(WAIT, provider.started.notified()).await.unwrap();

    let error = runtime
        .start_turn(Agent::default(), &session.id, "Second")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::TurnActive(id) if id == session.id));
    let messages = runtime.store().list_messages(&session.id).await.unwrap();
    assert_eq!(messages.len(), 1, "the rejected turn saved nothing");
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn turns_in_different_sessions_run_concurrently() {
    const SESSIONS: usize = 8;
    let provider = Arc::new(RendezvousProvider {
        barrier: Barrier::new(SESSIONS),
    });
    let runtime = memory_runtime(provider).await;
    let mut turns = Vec::new();
    for index in 0..SESSIONS {
        let session = runtime.create_session(None).await.unwrap();
        let turn = runtime
            .start_turn(Agent::default(), &session.id, format!("Task {index}"))
            .await
            .unwrap();
        turns.push(turn);
    }
    // Every provider call waits for all the others, so this only finishes if
    // the turns are in flight at the same time.
    for turn in turns {
        let outcome = timeout(WAIT, turn.outcome()).await.unwrap().unwrap();
        assert_eq!(outcome.run.status, RunStatus::Finished);
    }
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn dropping_a_handle_leaves_the_turn_running() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("conveyor.sqlite");
    let runtime = Runtime::new(
        Store::open(&path).await.unwrap(),
        Arc::new(FakeProvider::new(["Finished unobserved."])),
    );
    let session = runtime.create_session(None).await.unwrap();
    drop(
        runtime
            .start_turn(Agent::default(), &session.id, "Work in the background.")
            .await
            .unwrap(),
    );
    // Nobody awaits the turn; it still runs to completion on its own.
    timeout(WAIT, async {
        while runtime.is_active(&session.id) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    runtime.close().await.unwrap();

    let store = Store::open(&path).await.unwrap();
    let runs = store.list_runs(&session.id).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].status, RunStatus::Finished);
    store.close().await.unwrap();
}

#[tokio::test]
async fn closing_cancels_running_turns_before_closing_the_store() {
    let provider = GateProvider::new([None]);
    let runtime = memory_runtime(provider.clone()).await;
    let session = runtime.create_session(None).await.unwrap();
    let turn = runtime
        .start_turn(Agent::default(), &session.id, "Wait for shutdown.")
        .await
        .unwrap();
    timeout(WAIT, provider.started.notified()).await.unwrap();

    timeout(WAIT, runtime.close()).await.unwrap().unwrap();
    let outcome = turn.outcome().await.unwrap();
    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(outcome.run.error.as_deref(), Some(RUNTIME_SHUTDOWN_REASON));
    assert!(matches!(
        runtime.store().list_sessions(None).await,
        Err(StorageError::StoreClosed)
    ));

    runtime.close().await.unwrap();
    assert!(matches!(
        runtime.create_session(None).await,
        Err(Error::Closed)
    ));
    assert!(matches!(
        runtime
            .start_turn(Agent::default(), &session.id, "Late")
            .await,
        Err(Error::Closed)
    ));
    assert!(matches!(
        runtime.interrupt(&session.id, "Late"),
        Err(Error::Closed)
    ));
}

#[tokio::test]
async fn turns_require_content_and_an_active_session() {
    let runtime = memory_runtime(Arc::new(FakeProvider::default())).await;
    let session = runtime.create_session(None).await.unwrap();

    let error = runtime
        .start_turn(Agent::default(), &session.id, " \n\t")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::EmptyContent));

    let error = runtime
        .start_turn(Agent::default(), "ses_missing", "Hello")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::SessionNotFound(id) if id == "ses_missing"));

    let archived = Session {
        status: SessionStatus::Archived,
        ..Session::default()
    };
    runtime.store().save_session(&archived).await.unwrap();
    let error = runtime
        .start_turn(Agent::default(), &archived.id, "Hello")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::SessionNotActive(id) if id == archived.id));
    assert!(
        runtime
            .store()
            .list_messages(&session.id)
            .await
            .unwrap()
            .is_empty()
    );
    runtime.close().await.unwrap();
}
