use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use conveyor_agent::{Error, RunRequest, run_agent};
use conveyor_core::{
    BoxFuture, CancellationToken, Cancelled,
    models::{
        Agent, EventType, JsonObject, Message, MessageRole, RunStatus, Session, ToolCall,
        ToolPermission,
    },
    providers::{ProviderMessage, ProviderResponse, ToolDefinition, fake::FakeProvider},
    tools::{DynTool, Tool, ToolContext, ToolError, ToolOutput},
};
use conveyor_storage::Store;
use conveyor_tools::ToolRegistry;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

const NO_RESOLVER: &str = "Tool call denied because no approval resolver is configured";

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

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

fn agent(tools: &[&str]) -> Agent {
    Agent {
        tools: tools.iter().map(|name| (*name).to_owned()).collect(),
        ..Agent::default()
    }
}

/// One assistant reply requesting every call in `calls`.
fn batch(calls: &[ToolCall]) -> ProviderResponse {
    ProviderResponse {
        tool_calls: calls.to_vec(),
        ..ProviderResponse::tool_call(calls[0].clone())
    }
}

fn tool_messages(messages: &[Message]) -> Vec<&Message> {
    messages
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .collect()
}

#[derive(Deserialize)]
struct LookupArgs {
    key: String,
}

#[derive(Debug, thiserror::Error)]
enum LookupError {
    #[error("No entry for {0}")]
    Missing(String),
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}

/// A read tool. `"missing"` fails, `"wait"` waits for cancellation, and
/// `"release"` waits for [`Lookup::release`] and then returns.
#[derive(Clone, Default)]
struct Lookup {
    calls: Arc<AtomicUsize>,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

impl Tool for Lookup {
    const NAME: &'static str = "lookup";
    type Args = LookupArgs;
    type Output = Value;
    type Error = LookupError;

    fn description(&self) -> String {
        "Look up a key".into()
    }

    fn parameters(&self) -> JsonObject {
        object(json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: LookupError) -> ToolError {
        match error {
            LookupError::Cancelled(cancelled) => cancelled.into(),
            other => ToolError::failed(other),
        }
    }

    async fn call(&self, args: LookupArgs, context: &ToolContext) -> Result<Value, LookupError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        match args.key.as_str() {
            "missing" => Err(LookupError::Missing(args.key)),
            "wait" => Err(context.cancellation().cancelled().await.into()),
            "release" => {
                self.release.notified().await;
                Ok(json!({"key": "release", "value": 2}))
            }
            key => Ok(json!({"key": key, "value": 1})),
        }
    }
}

/// A tool with side effects, recording every call it actually runs.
#[derive(Clone)]
struct Effect {
    name: &'static str,
    permission: ToolPermission,
    calls: Arc<AtomicUsize>,
}

impl Effect {
    fn new(name: &'static str, permission: ToolPermission) -> Self {
        Self {
            name,
            permission,
            calls: Arc::default(),
        }
    }
}

// Implemented directly so one type can stand in for several named tools.
impl DynTool for Effect {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name,
            "Change the workspace",
            object(json!({"type": "object", "properties": {}})),
        )
    }

    fn permission(&self) -> ToolPermission {
        self.permission
    }

    fn parallel_safe(&self) -> bool {
        false
    }

    fn execute<'a>(
        &'a self,
        _arguments: JsonObject,
        _context: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("changed".into())
        })
    }
}

struct Tools {
    registry: ToolRegistry,
    lookup: Lookup,
    write: Effect,
    bash: Effect,
}

fn tools() -> Tools {
    let lookup = Lookup::default();
    let write = Effect::new("write_file", ToolPermission::Write);
    let bash = Effect::new("bash", ToolPermission::Dangerous);
    let registry = ToolRegistry::new()
        .with(lookup.clone())
        .unwrap()
        .with(write.clone())
        .unwrap()
        .with(bash.clone())
        .unwrap();
    Tools {
        registry,
        lookup,
        write,
        bash,
    }
}

#[tokio::test]
async fn allowed_tools_run_and_their_results_reach_the_model() {
    let (store, session) = seeded().await;
    let tools = tools();
    let call = ToolCall::new("lookup", object(json!({"key": "a"})));
    let provider = FakeProvider::new([
        ProviderResponse::tool_call(call.clone()),
        ProviderResponse::message("Found it."),
    ]);
    let agent = agent(&["lookup"]);
    let cancellation = CancellationToken::new();

    let outcome = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    )
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert_eq!(outcome.iterations, 2);
    assert_eq!(tools.lookup.calls.load(Ordering::SeqCst), 1);

    // Only the agent's own tools are advertised.
    let requests = provider.requests();
    let advertised: Vec<_> = requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(advertised, ["lookup"]);
    assert_eq!(requests[1].tools, requests[0].tools);

    let messages = store.list_messages(&session.id).await.unwrap();
    let results = tool_messages(&messages);
    assert_eq!(results.len(), 1);
    let result = results[0];
    assert_eq!(result.content, r#"{"key":"a","value":1}"#);
    assert_eq!(result.name.as_deref(), Some("lookup"));
    assert_eq!(result.tool_call_id.as_deref(), Some(call.id.as_str()));
    assert_eq!(result.run_id.as_deref(), Some(outcome.run.id.as_str()));
    assert_eq!(
        Value::Object(result.metadata.clone()),
        json!({"ok": true, "permission": "read", "content_type": "application/json"})
    );
    assert_eq!(
        requests[1].messages.last().cloned().unwrap(),
        ProviderMessage::tool_result(&call.id, "lookup", &result.content, false)
    );

    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [
            EventType::RunStarted,
            EventType::MessageCreated,
            EventType::ToolStarted,
            EventType::MessageCreated,
            EventType::ToolFinished,
            EventType::MessageCreated,
            EventType::RunFinished
        ]
    );
    let events = store
        .list_events(None, Some(&outcome.run.id))
        .await
        .unwrap();
    assert_eq!(
        Value::Object(events[2].payload.clone()),
        json!({"tool_call_id": call.id, "name": "lookup", "arguments": {"key": "a"}})
    );
    assert_eq!(events[4].message_id.as_deref(), Some(result.id.as_str()));
    assert_eq!(
        Value::Object(events[4].payload.clone()),
        json!({"tool_call_id": call.id, "name": "lookup", "ok": true})
    );
}

#[tokio::test]
async fn registered_tools_the_agent_does_not_list_are_unknown() {
    let (store, session) = seeded().await;
    let tools = tools();
    let call = ToolCall::new("lookup", object(json!({"key": "a"})));
    let provider = FakeProvider::new([ProviderResponse::tool_call(call.clone())]);
    let agent = agent(&[]);
    let cancellation = CancellationToken::new();

    let outcome = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    )
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert!(provider.requests()[0].tools.is_empty());
    assert_eq!(tools.lookup.calls.load(Ordering::SeqCst), 0);
    let messages = store.list_messages(&session.id).await.unwrap();
    assert_eq!(tool_messages(&messages)[0].content, "Unknown tool: lookup");
}

#[tokio::test]
async fn tools_that_need_approval_are_denied_without_running() {
    let (store, session) = seeded().await;
    let tools = tools();
    let write = ToolCall::new("write_file", JsonObject::new());
    let bash = ToolCall::new("bash", JsonObject::new());
    let provider = FakeProvider::new([batch(&[write.clone(), bash.clone()])]);
    let agent = agent(&["bash", "lookup", "write_file"]);
    let cancellation = CancellationToken::new();

    let outcome = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    )
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    assert_eq!(tools.write.calls.load(Ordering::SeqCst), 0);
    assert_eq!(tools.bash.calls.load(Ordering::SeqCst), 0);
    let messages = store.list_messages(&session.id).await.unwrap();
    let results = tool_messages(&messages);
    let correlated: Vec<_> = results
        .iter()
        .map(|message| message.tool_call_id.as_deref().unwrap())
        .collect();
    assert_eq!(correlated, [write.id.as_str(), bash.id.as_str()]);
    for result in results {
        assert_eq!(result.content, NO_RESOLVER);
        assert_eq!(
            Value::Object(result.metadata.clone()),
            json!({"ok": false, "policy_action": "deny"})
        );
    }
    let kinds = event_kinds(&store, &outcome.run.id).await;
    assert!(!kinds.contains(&EventType::ToolStarted));
}

#[tokio::test]
async fn failed_calls_are_reported_to_the_model_in_call_order() {
    let (store, session) = seeded().await;
    let tools = tools();
    let missing = ToolCall::new("lookup", object(json!({"key": "missing"})));
    let invalid = ToolCall::new("lookup", object(json!({"key": 7})));
    let found = ToolCall::new("lookup", object(json!({"key": "b"})));
    let provider = FakeProvider::new([
        batch(&[missing.clone(), invalid.clone(), found.clone()]),
        ProviderResponse::message("Two failed."),
    ]);
    let agent = agent(&["lookup"]);
    let cancellation = CancellationToken::new();

    let outcome = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    )
    .await
    .unwrap();

    assert_eq!(outcome.run.status, RunStatus::Finished);
    let resent = &provider.requests()[1].messages;
    assert_eq!(
        resent[resent.len() - 3..],
        [
            ProviderMessage::tool_result(&missing.id, "lookup", "No entry for missing", true),
            ProviderMessage::tool_result(
                &invalid.id,
                "lookup",
                "Invalid arguments for lookup: invalid type: integer `7`, expected a string",
                true,
            ),
            ProviderMessage::tool_result(&found.id, "lookup", r#"{"key":"b","value":1}"#, false),
        ]
    );
    let messages = store.list_messages(&session.id).await.unwrap();
    assert_eq!(
        Value::Object(tool_messages(&messages)[0].metadata.clone()),
        json!({"ok": false})
    );
}

#[tokio::test]
async fn cancellation_during_a_tool_call_cancels_the_rest_of_the_batch() {
    let (store, session) = seeded().await;
    let tools = tools();
    let waiting = ToolCall::new("lookup", object(json!({"key": "wait"})));
    let next = ToolCall::new("lookup", object(json!({"key": "a"})));
    let provider = FakeProvider::new([batch(&[waiting.clone(), next.clone()])]);
    let agent = agent(&["lookup"]);
    let cancellation = CancellationToken::new();

    let run = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    );
    let (outcome, ()) = tokio::join!(run, async {
        tools.lookup.started.notified().await;
        cancellation.cancel("Stopped by the user");
    });
    let outcome = outcome.unwrap();

    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(outcome.run.error.as_deref(), Some("Stopped by the user"));
    assert_eq!(tools.lookup.calls.load(Ordering::SeqCst), 1);
    let messages = store.list_messages(&session.id).await.unwrap();
    let results = tool_messages(&messages);
    assert_eq!(results.len(), 2);
    for (result, call) in results.iter().zip([&waiting, &next]) {
        assert_eq!(result.tool_call_id.as_deref(), Some(call.id.as_str()));
        assert_eq!(
            result.content,
            "Tool execution cancelled: Stopped by the user"
        );
    }
    assert_eq!(
        event_kinds(&store, &outcome.run.id).await,
        [
            EventType::RunStarted,
            EventType::MessageCreated,
            EventType::ToolStarted,
            EventType::MessageCreated,
            EventType::ToolCancelled,
            EventType::MessageCreated,
            EventType::ToolCancelled,
            EventType::RunCancelled
        ]
    );
}

#[tokio::test]
async fn a_call_that_finishes_as_cancellation_arrives_keeps_its_result() {
    let (store, session) = seeded().await;
    let tools = tools();
    let finishing = ToolCall::new("lookup", object(json!({"key": "release"})));
    let next = ToolCall::new("lookup", object(json!({"key": "a"})));
    let provider = FakeProvider::new([batch(&[finishing.clone(), next.clone()])]);
    let agent = agent(&["lookup"]);
    let cancellation = CancellationToken::new();

    let run = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    );
    // The tool finishes and the turn is cancelled before the run looks again,
    // so both are ready at once.
    let (outcome, ()) = tokio::join!(run, async {
        tools.lookup.started.notified().await;
        cancellation.cancel("Stopped after finishing");
        tools.lookup.release.notify_one();
    });
    let outcome = outcome.unwrap();

    assert_eq!(outcome.run.status, RunStatus::Cancelled);
    assert_eq!(tools.lookup.calls.load(Ordering::SeqCst), 1);
    let messages = store.list_messages(&session.id).await.unwrap();
    let results = tool_messages(&messages);
    assert_eq!(
        results[0].tool_call_id.as_deref(),
        Some(finishing.id.as_str())
    );
    assert_eq!(results[0].content, r#"{"key":"release","value":2}"#);
    assert_eq!(results[0].metadata["ok"], json!(true));
    assert_eq!(results[1].tool_call_id.as_deref(), Some(next.id.as_str()));
    assert_eq!(
        results[1].content,
        "Tool execution cancelled: Stopped after finishing"
    );
    let kinds = event_kinds(&store, &outcome.run.id).await;
    assert_eq!(
        kinds[2..5],
        [
            EventType::ToolStarted,
            EventType::MessageCreated,
            EventType::ToolFinished
        ]
    );
}

#[tokio::test]
async fn agents_cannot_list_tools_the_registry_lacks() {
    let (store, session) = seeded().await;
    let tools = tools();
    let provider = FakeProvider::default();
    let agent = agent(&["lookup", "web_search", "edit_file", "web_search"]);
    let cancellation = CancellationToken::new();

    let error = run_agent(
        RunRequest::new(&agent, &session, &provider, &store, &cancellation)
            .with_tools(&tools.registry, Path::new(".")),
    )
    .await
    .unwrap_err();

    assert!(matches!(&error, Error::UnknownTools(names) if names == &["edit_file", "web_search"]));
    assert!(store.list_runs(&session.id).await.unwrap().is_empty());
    assert!(provider.requests().is_empty());
}
