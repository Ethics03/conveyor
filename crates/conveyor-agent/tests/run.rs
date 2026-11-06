use std::{future, num::NonZeroU32, sync::Arc, time::Duration};

use conveyor_agent::{Error, RunRequest, run_agent};
use conveyor_core::{
    BoxFuture, CancellationToken,
    models::{Agent, EventType, JsonObject, Message, MessageRole, RunStatus, Session, ToolCall},
    providers::{
        FinishReason, ModelLimits, Provider, ProviderError, ProviderMessage, ProviderRequest,
        ProviderResponse, ReplayState, fake::FakeProvider,
    },
};
use conveyor_storage::Store;
use serde_json::{Value, json};
use tokio::sync::Notify;

async fn seeded() -> (Store, Session) {
    let store = Store::open(":memory:").await.unwrap();
    let session = Session::default();
    store.save_session(&session).await.unwrap();
    let message = Message::new(&session.id, MessageRole::User, "Complete the task.");
    store.save_message(&message).await.unwrap();
    (store, session)
}

async fn event_kinds(store: &Store, run_id: &str) -> Vec<EventType> {
    store
        .list_events(None, Some(run_id))
        .await
        .unwrap()
        .into_iter()
        .map(|event| event.kind)
        .collect()
}

fn roles(messages: &[Message]) -> Vec<MessageRole> {
    messages.iter().map(|message| message.role).collect()
}

/// Signals when a request starts, then never answers.
#[derive(Debug, Default)]
struct WaitingProvider {
    started: Notify,
}

impl Provider for WaitingProvider {
    fn name(&self) -> &str {
        "waiting"
    }

    fn model_limits(&self, _model: Option<&str>) -> ModelLimits {
        ModelLimits::new(200_000, 4_096).unwrap()
    }

    fn generate(
        &self,
        _request: ProviderRequest,
    ) -> BoxFuture<'_, Result<ProviderResponse, ProviderError>> {
        Box::pin(async move {
            self.started.notify_one();
            future::pending().await
        })
    }
}

#[tokio::test]
async fn a_plain_response_finishes_the_run() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::new(["Done."]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();

    let outcome = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert_eq!(outcome.run.error, None);
    assert_eq!(outcome.iterations, 1);
    assert_eq!(outcome.final_message.as_ref().unwrap().content, "Done.");
    let messages = store.list_messages(&session.id).await.unwrap();
    assert_eq!(
        roles(&messages),
        [MessageRole::User, MessageRole::Assistant]
    );
    assert_eq!(messages[1].run_id.as_deref(), Some(outcome.run.id.as_str()));
    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [
            EventType::RunStarted,
            EventType::MessageCreated,
            EventType::RunFinished
        ]
    );
    let stored = store.get_run(&outcome.run.id).await.unwrap().unwrap();
    assert_eq!(stored, outcome.run);
}

#[tokio::test]
async fn requests_carry_instructions_temporal_context_model_and_output_limit() {
    let (store, session) = seeded().await;
    let limits = ModelLimits::new(10_000, 512).unwrap();
    let provider = FakeProvider::new(["Done."]).with_model_limits(limits);
    let agent = Agent {
        instructions: "Be precise.".into(),
        model: Some("claude-test".into()),
        ..Agent::default()
    };
    let cancellation = CancellationToken::new();
    run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();

    let request = &provider.requests()[0];
    assert_eq!(request.model.as_deref(), Some("claude-test"));
    assert_eq!(request.max_tokens, Some(512));
    assert!(request.tools.is_empty());
    assert_eq!(request.messages[0], ProviderMessage::system("Be precise."));
    assert!(
        request.messages[1]
            .content()
            .starts_with("Temporal context (UTC):\n- Session created at: ")
    );
    let user = request.messages[2].content();
    assert!(user.starts_with("Message timestamp (UTC): "), "{user}");
    assert!(user.ends_with("\n\nComplete the task."), "{user}");
}

#[tokio::test]
async fn cancellation_before_the_provider_call_sends_nothing() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::new(["Should not be requested"]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    cancellation.cancel("Stopped from the client");

    let outcome = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(
        outcome.run.error.as_deref(),
        Some("Stopped from the client")
    );
    assert!(outcome.final_message.is_none());
    assert!(provider.requests().is_empty());
    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [EventType::RunStarted, EventType::RunCancelled]
    );
}

#[tokio::test]
async fn cancellation_interrupts_a_provider_call_in_flight() {
    let (store, session) = seeded().await;
    let provider = Arc::new(WaitingProvider::default());
    let cancellation = CancellationToken::new();
    let run = {
        let (provider, cancellation, store) = (Arc::clone(&provider), cancellation.clone(), store);
        tokio::spawn(async move {
            let agent = Agent::default();
            let outcome = run_agent(RunRequest::new(
                &agent,
                &session,
                provider.as_ref(),
                &store,
                &cancellation,
            ))
            .await;
            (outcome, store)
        })
    };
    provider.started.notified().await;
    cancellation.cancel("Stop provider");

    let (outcome, store) = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap();
    let outcome = outcome.unwrap();
    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(outcome.run.error.as_deref(), Some("Stop provider"));
    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [EventType::RunStarted, EventType::RunCancelled]
    );
}

#[tokio::test]
async fn incomplete_responses_fail_the_run_and_keep_the_partial_message() {
    let cases = [
        (
            FinishReason::MaxTokens,
            "Provider response stopped at a token limit",
        ),
        (FinishReason::Refusal, "Provider refused the request"),
        (
            FinishReason::Pause,
            "Provider paused without a resumable tool call",
        ),
        (
            FinishReason::Unknown,
            "Provider returned an unknown finish reason",
        ),
        (
            FinishReason::ToolUse,
            "Provider returned tool_use without any tool calls",
        ),
    ];
    for (finish_reason, expected) in cases {
        let (store, session) = seeded().await;
        let provider = FakeProvider::new([ProviderResponse {
            finish_reason,
            ..ProviderResponse::message("Partial response")
        }]);
        let agent = Agent::default();
        let cancellation = CancellationToken::new();
        let outcome = run_agent(RunRequest::new(
            &agent,
            &session,
            &provider,
            &store,
            &cancellation,
        ))
        .await
        .unwrap();

        assert_eq!(outcome.run.status, RunStatus::Failed);
        assert_eq!(outcome.run.error.as_deref(), Some(expected));
        assert_eq!(outcome.final_message.unwrap().content, "Partial response");
        assert_eq!(
            event_kinds(&store, &outcome.run.id).await,
            [
                EventType::RunStarted,
                EventType::MessageCreated,
                EventType::RunFailed
            ]
        );
    }
}

#[tokio::test]
async fn tool_calls_with_the_wrong_finish_reason_fail_the_run() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::new([ProviderResponse {
        finish_reason: FinishReason::Stop,
        ..ProviderResponse::tool_call(ToolCall::new("read_file", JsonObject::new()))
    }]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let outcome = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();
    assert_eq!(
        outcome.run.error.as_deref(),
        Some("Provider returned tool calls with finish reason: stop")
    );
}

#[tokio::test]
async fn unknown_tool_calls_get_correlated_denials_and_the_run_continues() {
    let (store, session) = seeded().await;
    let call = ToolCall::new("read_file", JsonObject::new());
    let provider = FakeProvider::new([
        ProviderResponse::tool_call(call.clone()),
        ProviderResponse::message("Done without tools."),
    ]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let outcome = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert_eq!(outcome.iterations, 2);
    let messages = store.list_messages(&session.id).await.unwrap();
    assert_eq!(
        roles(&messages),
        [
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Tool,
            MessageRole::Assistant
        ]
    );
    let denial = &messages[2];
    assert_eq!(denial.content, "Unknown tool: read_file");
    assert_eq!(denial.tool_call_id.as_deref(), Some(call.id.as_str()));
    assert_eq!(
        Value::Object(denial.metadata.clone()),
        json!({"ok": false, "policy_action": "deny"})
    );
    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [
            EventType::RunStarted,
            EventType::MessageCreated,
            EventType::MessageCreated,
            EventType::MessageCreated,
            EventType::RunFinished
        ]
    );
    let resent = provider.requests()[1].messages.last().cloned().unwrap();
    assert_eq!(
        resent,
        ProviderMessage::tool_result(&call.id, "read_file", "Unknown tool: read_file", true)
    );
}

#[tokio::test]
async fn the_iteration_limit_fails_the_run() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::new([ProviderResponse::tool_call(ToolCall::new(
        "echo",
        JsonObject::new(),
    ))]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let mut request = RunRequest::new(&agent, &session, &provider, &store, &cancellation);
    request.max_iterations = NonZeroU32::new(1).unwrap();
    let outcome = run_agent(request).await.unwrap();

    assert_eq!(outcome.run.status, RunStatus::Failed);
    assert_eq!(
        outcome.run.error.as_deref(),
        Some("Maximum iterations exceeded: 1")
    );
    assert_eq!(outcome.iterations, 1);
    let kinds = event_kinds(&store, &outcome.run.id).await;
    assert_eq!(kinds.last(), Some(&EventType::RunFailed));
}

#[tokio::test]
async fn provider_failures_are_recorded_on_the_run_and_returned() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::with_results([Err(ProviderError::InvalidResponse(
        "provider unavailable".into(),
    ))]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let error = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap_err();

    let Error::Provider { run_id, .. } = &error else {
        panic!("expected a provider error, got {error:?}");
    };
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_eq!(run.status, RunStatus::Failed);
    assert_eq!(
        run.error.as_deref(),
        Some("Provider request failed: Invalid provider response: provider unavailable")
    );
    assert_eq!(
        event_kinds(&store, run_id).await.last(),
        Some(&EventType::RunFailed)
    );
}

#[tokio::test]
async fn invalid_inputs_are_rejected_before_a_run_is_created() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::default();
    let cancellation = CancellationToken::new();
    let agent = Agent {
        tools: vec!["write_file".into(), "bash".into(), "bash".into()],
        ..Agent::default()
    };
    let error = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap_err();
    assert!(matches!(&error, Error::UnknownTools(names) if names == &["bash", "write_file"]));
    assert_eq!(error.to_string(), "Unknown tools: bash, write_file");

    let empty = Session::default();
    store.save_session(&empty).await.unwrap();
    let agent = Agent::default();
    let error = run_agent(RunRequest::new(
        &agent,
        &empty,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap_err();
    assert!(matches!(error, Error::EmptyTranscript));
    assert!(store.list_runs(&session.id).await.unwrap().is_empty());
    assert!(store.list_runs(&empty.id).await.unwrap().is_empty());
}

#[tokio::test]
async fn assistant_metadata_keeps_provider_details_and_replay_state() {
    let (store, session) = seeded().await;
    let mut raw = JsonObject::new();
    raw.insert("provider".into(), "anthropic".into());
    raw.insert("response_id".into(), "msg_1".into());
    let state = ReplayState {
        provider: "anthropic".into(),
        items: vec![match json!({"type": "compaction", "content": "summary"}) {
            Value::Object(item) => item,
            _ => unreachable!(),
        }],
    };
    let provider = FakeProvider::new([ProviderResponse {
        raw,
        replay_state: Some(state.clone()),
        ..ProviderResponse::message("Compacted.")
    }]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let outcome = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();

    let message = outcome.final_message.unwrap();
    assert_eq!(
        Value::Object(message.metadata),
        json!({
            "provider": "anthropic",
            "response_id": "msg_1",
            "finish_reason": "stop",
            "provider_replay_state": state.to_metadata(),
        })
    );
    let events = store
        .list_events(None, Some(&outcome.run.id))
        .await
        .unwrap();
    let compacted = events
        .iter()
        .find(|event| event.kind == EventType::ContextCompacted)
        .unwrap();
    assert_eq!(compacted.message_id.as_deref(), Some(message.id.as_str()));
    assert_eq!(
        Value::Object(compacted.payload.clone()),
        json!({"provider": "anthropic", "item_count": 1})
    );
}

#[tokio::test]
async fn child_runs_record_their_parent() {
    let (store, session) = seeded().await;
    let provider = FakeProvider::new(["parent", "child"]);
    let agent = Agent::default();
    let cancellation = CancellationToken::new();
    let parent = run_agent(RunRequest::new(
        &agent,
        &session,
        &provider,
        &store,
        &cancellation,
    ))
    .await
    .unwrap();
    let mut request = RunRequest::new(&agent, &session, &provider, &store, &cancellation);
    request.parent_run_id = Some(&parent.run.id);
    let child = run_agent(request).await.unwrap();
    let stored = store.get_run(&child.run.id).await.unwrap().unwrap();
    assert_eq!(
        stored.parent_run_id.as_deref(),
        Some(parent.run.id.as_str())
    );
}
