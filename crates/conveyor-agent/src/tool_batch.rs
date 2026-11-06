//! Persistence of tool-call results that a run produces without executing tools.

use std::collections::HashSet;

use conveyor_core::models::{EventType, Message, MessageRole, Run, ToolCall};
use conveyor_storage::Store;
use serde_json::json;

use crate::{
    Result,
    run::{event, payload},
};

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
