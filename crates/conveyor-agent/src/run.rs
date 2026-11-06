//! One agent run: provider iterations, run transitions, and their persistence.

use std::num::NonZeroU32;

use conveyor_core::{
    CancellationToken, Cancelled,
    models::{
        Agent, Event, EventType, JsonObject, Message, MessageRole, Run, RunOutcome, RunStatus,
        Session, now,
    },
    providers::{FinishReason, Provider, ProviderRequest, ProviderResponse, ReplayState},
};
use conveyor_storage::{Error as StorageError, Store};
use serde_json::{Value, json};

use crate::{
    Error, Result,
    context::build_provider_messages,
    error::error_chain,
    tool_batch::{deny_tool_call, save_cancelled_tool_results},
};

/// Provider iterations allowed per run unless configured otherwise.
pub const DEFAULT_MAX_ITERATIONS: NonZeroU32 = NonZeroU32::new(20).unwrap();

/// Everything one run needs. Build it with [`RunRequest::new`].
#[derive(Debug, Clone, Copy)]
pub struct RunRequest<'a> {
    pub agent: &'a Agent,
    pub session: &'a Session,
    pub provider: &'a dyn Provider,
    pub store: &'a Store,
    pub cancellation: &'a CancellationToken,
    pub parent_run_id: Option<&'a str>,
    pub max_iterations: NonZeroU32,
}

impl<'a> RunRequest<'a> {
    pub fn new(
        agent: &'a Agent,
        session: &'a Session,
        provider: &'a dyn Provider,
        store: &'a Store,
        cancellation: &'a CancellationToken,
    ) -> Self {
        Self {
            agent,
            session,
            provider,
            store,
            cancellation,
            parent_run_id: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

/// Run `agent` over the session's stored transcript until it answers, fails,
/// or is cancelled.
///
/// The run and every message and event it produces are persisted as they
/// happen. Cancellation is checked before and after each provider call and
/// interrupts a call in flight; the outcome is then `cancelled`, with a
/// cancelled result for each unanswered tool call. Incomplete provider
/// responses and the iteration limit end the run as `failed`.
///
/// Returns an error without creating a run when the agent names unavailable
/// tools or the session has no messages. Provider and storage failures after
/// the run starts are recorded on the run, then returned.
pub async fn run_agent(request: RunRequest<'_>) -> Result<RunOutcome> {
    if !request.agent.tools.is_empty() {
        let mut names = request.agent.tools.clone();
        names.sort();
        names.dedup();
        return Err(Error::UnknownTools(names));
    }
    let mut transcript = request.store.list_messages(&request.session.id).await?;
    if transcript.is_empty() {
        return Err(Error::EmptyTranscript);
    }

    let mut run = Run::new(&request.session.id, &request.agent.id);
    run.parent_run_id = request.parent_run_id.map(str::to_owned);
    run.status = RunStatus::Running;
    let mut started = event(EventType::RunStarted, &run);
    started.payload = payload(json!({"agent_id": request.agent.id}));
    request.store.save_run_with_events(&run, &[started]).await?;

    let mut progress = Progress {
        run,
        final_message: None,
        iterations: 0,
    };
    match iterate(&request, &mut progress, &mut transcript).await {
        Ok(None) => finish(request.store, progress).await,
        Ok(Some(error)) => fail(request.store, progress, error).await,
        Err(Stop::Cancelled(cancelled)) => cancel(request.store, progress, cancelled).await,
        Err(Stop::Failed(error)) => {
            // Record the failure if storage still works; the original error wins.
            let _ = fail(request.store, progress, error_chain(&error)).await;
            Err(error)
        }
    }
}

struct Progress {
    run: Run,
    final_message: Option<Message>,
    iterations: u32,
}

enum Stop {
    Cancelled(Cancelled),
    Failed(Error),
}

impl From<Cancelled> for Stop {
    fn from(cancelled: Cancelled) -> Self {
        Self::Cancelled(cancelled)
    }
}

impl From<Error> for Stop {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

impl From<StorageError> for Stop {
    fn from(error: StorageError) -> Self {
        Self::Failed(error.into())
    }
}

/// Returns `Some(reason)` when the run should end as failed.
async fn iterate(
    request: &RunRequest<'_>,
    progress: &mut Progress,
    transcript: &mut Vec<Message>,
) -> std::result::Result<Option<String>, Stop> {
    let cancellation = request.cancellation;
    for iteration in 1..=request.max_iterations.get() {
        progress.iterations = iteration;
        cancellation.check()?;
        let provider_request = provider_request(request, &progress.run, transcript)?;
        let response = tokio::select! {
            biased;
            cancelled = cancellation.cancelled() => return Err(cancelled.into()),
            response = request.provider.generate(provider_request) => {
                response.map_err(|source| Error::Provider {
                    run_id: progress.run.id.clone(),
                    source,
                })?
            }
        };
        cancellation.check()?;

        let assistant = save_assistant_message(request.store, &progress.run, &response).await?;
        transcript.push(assistant.clone());
        progress.final_message = Some(assistant.clone());
        if let Some(error) = response_error(&response) {
            return Ok(Some(error));
        }
        if assistant.tool_calls.is_empty() {
            return Ok(None);
        }
        // No tools are registered yet, so every call is denied as unknown.
        for call in &assistant.tool_calls {
            cancellation.check()?;
            let reason = format!("Unknown tool: {}", call.name);
            let denial = deny_tool_call(request.store, &progress.run, call, &reason).await?;
            transcript.push(denial);
        }
    }
    Ok(Some(format!(
        "Maximum iterations exceeded: {}",
        request.max_iterations
    )))
}

fn provider_request(
    request: &RunRequest<'_>,
    run: &Run,
    transcript: &[Message],
) -> Result<ProviderRequest> {
    let messages =
        build_provider_messages(request.agent, transcript, Some((request.session, run)))?;
    let limits = request
        .provider
        .model_limits(request.agent.model.as_deref());
    let mut provider_request = ProviderRequest::new(messages);
    provider_request.model = request.agent.model.clone();
    provider_request.max_tokens = Some(limits.max_output_tokens());
    Ok(provider_request)
}

async fn save_assistant_message(
    store: &Store,
    run: &Run,
    response: &ProviderResponse,
) -> Result<Message> {
    let mut message = Message::new(&run.session_id, MessageRole::Assistant, &response.content);
    message.run_id = Some(run.id.clone());
    message.tool_calls = response.tool_calls.clone();
    message.metadata = response.raw.clone();
    message.metadata.insert(
        "finish_reason".into(),
        response.finish_reason.as_str().into(),
    );
    if let Some(state) = &response.replay_state {
        message
            .metadata
            .insert(ReplayState::METADATA_KEY.into(), state.to_metadata());
    }

    let mut created = event(EventType::MessageCreated, run);
    created.message_id = Some(message.id.clone());
    created.payload = payload(json!({
        "role": "assistant",
        "tool_call_count": message.tool_calls.len(),
    }));
    let mut events = vec![created];
    if let Some(state) = &response.replay_state {
        let compactions = state
            .items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction"))
            .count();
        if compactions > 0 {
            let mut compacted = event(EventType::ContextCompacted, run);
            compacted.message_id = Some(message.id.clone());
            compacted.payload = payload(json!({
                "provider": state.provider,
                "item_count": compactions,
            }));
            events.push(compacted);
        }
    }
    store.save_message_with_events(&message, &events).await?;
    Ok(message)
}

/// Why a response cannot continue the run, if it cannot.
fn response_error(response: &ProviderResponse) -> Option<String> {
    let reason = response.finish_reason;
    if !response.tool_calls.is_empty() {
        return (reason != FinishReason::ToolUse).then(|| {
            format!(
                "Provider returned tool calls with finish reason: {}",
                reason.as_str()
            )
        });
    }
    let error = match reason {
        FinishReason::Stop => None,
        FinishReason::ToolUse => Some("Provider returned tool_use without any tool calls"),
        FinishReason::MaxTokens => Some("Provider response stopped at a token limit"),
        FinishReason::Refusal => Some("Provider refused the request"),
        FinishReason::Pause => Some("Provider paused without a resumable tool call"),
        FinishReason::Unknown => Some("Provider returned an unknown finish reason"),
    };
    error.map(str::to_owned)
}

async fn finish(store: &Store, mut progress: Progress) -> Result<RunOutcome> {
    progress.run.status = RunStatus::Finished;
    progress.run.error = None;
    progress.run.updated_at = now();
    let mut finished = event(EventType::RunFinished, &progress.run);
    finished.message_id = progress
        .final_message
        .as_ref()
        .map(|message| message.id.clone());
    finished.payload = payload(json!({"iterations": progress.iterations}));
    store
        .save_run_with_events(&progress.run, &[finished])
        .await?;
    Ok(outcome(progress))
}

async fn fail(store: &Store, mut progress: Progress, error: String) -> Result<RunOutcome> {
    progress.run.status = RunStatus::Failed;
    progress.run.updated_at = now();
    let mut failed = event(EventType::RunFailed, &progress.run);
    failed.message_id = progress
        .final_message
        .as_ref()
        .map(|message| message.id.clone());
    failed.payload = payload(json!({"error": error, "iterations": progress.iterations}));
    progress.run.error = Some(error);
    store.save_run_with_events(&progress.run, &[failed]).await?;
    Ok(outcome(progress))
}

async fn cancel(store: &Store, mut progress: Progress, cancelled: Cancelled) -> Result<RunOutcome> {
    let reason = cancelled.reason();
    if let Some(assistant) = &progress.final_message {
        save_cancelled_tool_results(store, &progress.run, assistant, reason).await?;
    }
    progress.run.status = RunStatus::Cancelled;
    progress.run.error = Some(reason.to_owned());
    progress.run.updated_at = now();
    let mut event = event(EventType::RunCancelled, &progress.run);
    event.message_id = progress
        .final_message
        .as_ref()
        .map(|message| message.id.clone());
    event.payload = payload(json!({"reason": reason, "iterations": progress.iterations}));
    store.save_run_with_events(&progress.run, &[event]).await?;
    Ok(outcome(progress))
}

fn outcome(progress: Progress) -> RunOutcome {
    RunOutcome {
        run: progress.run,
        final_message: progress.final_message,
        iterations: progress.iterations as usize,
    }
}

/// An event scoped to `run` and its session.
pub(crate) fn event(kind: EventType, run: &Run) -> Event {
    let mut event = Event::new(kind);
    event.session_id = Some(run.session_id.clone());
    event.run_id = Some(run.id.clone());
    event
}

/// The object inside a `json!({...})` literal.
pub(crate) fn payload(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        _ => JsonObject::new(),
    }
}
