//! Provider request construction from the stored transcript.

use chrono::Timelike;
use conveyor_core::{
    models::{Agent, Message, MessageRole, Run, Session, Timestamp},
    providers::{ProviderMessage, ReplayState},
};
use serde_json::Value;

use crate::{Error, Result};

/// Convert a stored transcript into provider messages.
///
/// Non-empty agent instructions come first as a system message. With
/// `temporal` context, a second system message records when the session was
/// created and each user message is prefixed with its timestamp; both depend
/// only on stored data, so the prefix stays the same across runs.
pub fn build_provider_messages(
    agent: &Agent,
    messages: &[Message],
    temporal: Option<(&Session, &Run)>,
) -> Result<Vec<ProviderMessage>> {
    let mut converted = Vec::with_capacity(messages.len() + 2);
    if !agent.instructions.is_empty() {
        converted.push(ProviderMessage::system(agent.instructions.clone()));
    }
    if let Some((session, run)) = temporal {
        if run.session_id != session.id {
            return Err(Error::RunSessionMismatch);
        }
        converted.push(ProviderMessage::system(format!(
            "Temporal context (UTC):\n- Session created at: {}",
            isoformat(&session.created_at)
        )));
    }
    for message in messages {
        converted.push(provider_message(message, temporal.is_some())?);
    }
    Ok(converted)
}

fn provider_message(message: &Message, include_timestamp: bool) -> Result<ProviderMessage> {
    Ok(match message.role {
        MessageRole::System => ProviderMessage::system(message.content.clone()),
        MessageRole::User if include_timestamp => ProviderMessage::user(format!(
            "Message timestamp (UTC): {}\n\n{}",
            isoformat(&message.created_at),
            message.content
        )),
        MessageRole::User => ProviderMessage::user(message.content.clone()),
        MessageRole::Assistant => ProviderMessage::Assistant {
            content: message.content.clone(),
            tool_calls: message.tool_calls.clone(),
            replay_state: message
                .metadata
                .get(ReplayState::METADATA_KEY)
                .and_then(ReplayState::from_metadata),
        },
        MessageRole::Tool => ProviderMessage::Tool {
            tool_call_id: message
                .tool_call_id
                .clone()
                .ok_or_else(|| Error::MissingToolCallId(message.id.clone()))?,
            name: message.name.clone().unwrap_or_else(|| "tool".into()),
            content: message.content.clone(),
            is_error: message.metadata.get("ok") == Some(&Value::Bool(false)),
        },
    })
}

/// ISO 8601 with the stored offset, and microseconds only when present.
fn isoformat(timestamp: &Timestamp) -> String {
    let format = if timestamp.nanosecond() == 0 {
        "%Y-%m-%dT%H:%M:%S%:z"
    } else {
        "%Y-%m-%dT%H:%M:%S%.6f%:z"
    };
    timestamp.format(format).to_string()
}
