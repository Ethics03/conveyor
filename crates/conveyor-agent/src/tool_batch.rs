//! Tool-call decisions, execution, and the persistence of their results.

use std::collections::HashSet;

use conveyor_core::{
    models::{EventType, Message, MessageRole, Run, ToolCall, ToolPermission},
    tools::ToolContext,
};
use conveyor_storage::Store;
use conveyor_tools::ToolRegistry;
use serde_json::json;

use crate::{
    Result,
    run::{Stop, event, payload},
};

/// The content of a denial for a tool that needs approval.
pub(crate) const NO_APPROVAL_RESOLVER: &str =
    "Tool call denied because no approval resolver is configured";

/// The tools one run may call, and the context they execute with.
pub(crate) struct Toolbox {
    /// The registry restricted to the agent's tools.
    pub(crate) registry: ToolRegistry,
    pub(crate) context: ToolContext,
}

/// What happens to one requested call.
pub(crate) enum Decision<'t> {
    Run(&'t Toolbox),
    Deny(String),
}

/// Decide every call in a batch before any of them runs.
///
/// Calls to tools outside the run's toolbox are denied as unknown. Read tools
/// run; write and dangerous tools need an approval, and with no way to ask
/// for one yet, they are denied.
pub(crate) fn preflight<'t>(toolbox: Option<&'t Toolbox>, calls: &[ToolCall]) -> Vec<Decision<'t>> {
    calls
        .iter()
        .map(|call| {
            let permission = toolbox
                .and_then(|toolbox| Some((toolbox, toolbox.registry.permission(&call.name)?)));
            match permission {
                None => Decision::Deny(format!("Unknown tool: {}", call.name)),
                Some((toolbox, ToolPermission::Read)) => Decision::Run(toolbox),
                Some((_, ToolPermission::Write | ToolPermission::Dangerous)) => {
                    Decision::Deny(NO_APPROVAL_RESOLVER.into())
                }
            }
        })
        .collect()
}

/// Execute one allowed call and save its result.
///
/// A result that is ready when cancellation arrives is saved as finished;
/// otherwise cancellation stops the call and is returned, leaving the
/// cancelled result to the run's cancellation path.
pub(crate) async fn execute_tool_call(
    store: &Store,
    run: &Run,
    toolbox: &Toolbox,
    call: &ToolCall,
) -> std::result::Result<Message, Stop> {
    let mut started = event(EventType::ToolStarted, run);
    started.payload = payload(json!({
        "tool_call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    }));
    store.append_event(&started).await?;

    let result = tokio::select! {
        biased;
        result = toolbox.registry.execute(call, &toolbox.context) => result?,
        cancelled = toolbox.context.cancellation().cancelled() => return Err(cancelled.into()),
    };

    let mut message = tool_message(run, call, &result.content);
    message.metadata = result.metadata;
    message.metadata.insert("ok".into(), result.ok.into());
    let mut created = event(EventType::MessageCreated, run);
    created.message_id = Some(message.id.clone());
    created.payload = payload(json!({"role": "tool", "name": call.name, "ok": result.ok}));
    let mut finished = event(EventType::ToolFinished, run);
    finished.message_id = Some(message.id.clone());
    finished.payload = payload(json!({
        "tool_call_id": call.id,
        "name": call.name,
        "ok": result.ok,
    }));
    store
        .save_message_with_events(&message, &[created, finished])
        .await?;
    Ok(message)
}

/// Save a policy denial as the call's correlated tool result.
pub(crate) async fn deny_tool_call(
    store: &Store,
    run: &Run,
    call: &ToolCall,
    reason: &str,
) -> Result<Message> {
    let mut message = tool_message(run, call, reason);
    message.metadata = payload(json!({"ok": false, "policy_action": "deny"}));
    let mut created = event(EventType::MessageCreated, run);
    created.message_id = Some(message.id.clone());
    created.payload = payload(json!({
        "role": "tool",
        "name": call.name,
        "ok": false,
        "policy_action": "deny",
    }));
    store.save_message_with_events(&message, &[created]).await?;
    Ok(message)
}

/// Give every call in `assistant` without a stored result a cancelled one.
pub(crate) async fn save_cancelled_tool_results(
    store: &Store,
    run: &Run,
    assistant: &Message,
    reason: &str,
) -> Result<()> {
    if assistant.tool_calls.is_empty() {
        return Ok(());
    }
    let answered: HashSet<String> = store
        .list_messages(&run.session_id)
        .await?
        .into_iter()
        .filter(|message| {
            message.role == MessageRole::Tool && message.run_id.as_deref() == Some(&run.id)
        })
        .filter_map(|message| message.tool_call_id)
        .collect();
    for call in &assistant.tool_calls {
        if answered.contains(&call.id) {
            continue;
        }
        let mut message = tool_message(run, call, &format!("Tool execution cancelled: {reason}"));
        message.metadata = payload(json!({"ok": false, "cancelled": true}));
        let mut created = event(EventType::MessageCreated, run);
        created.message_id = Some(message.id.clone());
        created.payload = payload(json!({"role": "tool", "name": call.name, "ok": false}));
        let mut cancelled = event(EventType::ToolCancelled, run);
        cancelled.message_id = Some(message.id.clone());
        cancelled.payload = payload(json!({
            "tool_call_id": call.id,
            "name": call.name,
            "reason": reason,
        }));
        store
            .save_message_with_events(&message, &[created, cancelled])
            .await?;
    }
    Ok(())
}

fn tool_message(run: &Run, call: &ToolCall, content: &str) -> Message {
    let mut message = Message::new(&run.session_id, MessageRole::Tool, content);
    message.run_id = Some(run.id.clone());
    message.name = Some(call.name.clone());
    message.tool_call_id = Some(call.id.clone());
    message
}
