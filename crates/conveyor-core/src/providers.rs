//! Provider contracts, neutral request and response types, and configuration.
//!
//! A [`Provider`] turns one [`ProviderRequest`] into one [`ProviderResponse`].
//! Adapters own their wire formats; these types stay provider-neutral so
//! context planning and orchestration never depend on a particular API.

pub mod fake;

use std::{error::Error as StdError, fmt::Debug, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    BoxFuture, Error, Result,
    models::{JsonObject, MessageRole, ToolCall, string_enum},
};

string_enum! {
    /// Why a provider stopped generating, normalized across providers.
    pub enum FinishReason {
        Stop => "stop", ToolUse => "tool_use", MaxTokens => "max_tokens",
        Refusal => "refusal", Pause => "pause", Unknown => "unknown"
    }
}

/// A model's token limits, validated so output always fits inside the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLimits {
    context_window_tokens: u32,
    max_output_tokens: u32,
}

impl ModelLimits {
    /// Validate limits: both positive, with output smaller than the window.
    pub const fn new(context_window_tokens: u32, max_output_tokens: u32) -> Result<Self> {
        if context_window_tokens == 0 {
            return Err(Error::EmptyContextWindow);
        }
        if max_output_tokens == 0 {
            return Err(Error::EmptyOutputLimit);
        }
        if max_output_tokens >= context_window_tokens {
            return Err(Error::OutputExceedsContextWindow);
        }
        Ok(Self {
            context_window_tokens,
            max_output_tokens,
        })
    }

    /// Total tokens the model accepts, including the reserved output.
    pub const fn context_window_tokens(self) -> u32 {
        self.context_window_tokens
    }

    /// Maximum tokens the model may generate in one response.
    pub const fn max_output_tokens(self) -> u32 {
        self.max_output_tokens
    }
}

/// A tool as advertised to the model: its name, description, and JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments, normally `{"type": "object", ...}`.
    pub parameters: JsonObject,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonObject,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Opaque, provider-owned items needed to continue a conversation, such as
/// Anthropic compaction blocks. Only the owning provider interprets `items`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayState {
    pub provider: String,
    pub items: Vec<JsonObject>,
}

impl ReplayState {
    /// The message metadata key under which replay state is persisted.
    pub const METADATA_KEY: &str = "provider_replay_state";

    /// Encode as the persisted `{"provider": ..., "items": [...]}` object.
    pub fn to_metadata(&self) -> Value {
        let mut value = Map::new();
        value.insert("provider".into(), Value::String(self.provider.clone()));
        let items = self.items.iter().cloned().map(Value::Object).collect();
        value.insert("items".into(), Value::Array(items));
        Value::Object(value)
    }

    /// Decode persisted metadata, returning `None` for any malformed shape.
    pub fn from_metadata(value: &Value) -> Option<Self> {
        Self::deserialize(value).ok()
    }
}

/// One provider-neutral conversation message.
///
/// Serializes with a `role` tag using the [`MessageRole`] spellings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ProviderMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay_state: Option<ReplayState>,
    },
    /// The result of one tool call, correlated by `tool_call_id`.
    Tool {
        tool_call_id: String,
        name: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl ProviderMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// A plain assistant reply without tool calls or replay state.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls: Vec::new(),
            replay_state: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: content.into(),
            is_error,
        }
    }

    pub fn role(&self) -> MessageRole {
        match self {
            Self::System { .. } => MessageRole::System,
            Self::User { .. } => MessageRole::User,
            Self::Assistant { .. } => MessageRole::Assistant,
            Self::Tool { .. } => MessageRole::Tool,
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Self::System { content }
            | Self::User { content }
            | Self::Assistant { content, .. }
            | Self::Tool { content, .. } => content,
        }
    }
}

/// One generation request. Unset options use the provider's defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub messages: Vec<ProviderMessage>,
    pub tools: Vec<ToolDefinition>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// The local context planner's compaction trigger, in input tokens.
    ///
    /// Providers with native compaction use it to compact before the local
    /// fallback would; others ignore it.
    pub compaction_trigger_tokens: Option<u32>,
}

impl ProviderRequest {
    /// A request for `messages` with no tools and default options.
    pub fn new(messages: Vec<ProviderMessage>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            model: None,
            temperature: None,
            max_tokens: None,
            compaction_trigger_tokens: None,
        }
    }
}

/// One generated response, normalized across providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    /// Provider details persisted with the assistant message, such as the
    /// provider name, model, response ID, and usage.
    pub raw: JsonObject,
    pub replay_state: Option<ReplayState>,
}

impl ProviderResponse {
    /// A text reply that finished normally.
    pub fn message(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            raw: JsonObject::new(),
            replay_state: None,
        }
    }

    /// A reply requesting one tool call, finishing with [`FinishReason::ToolUse`].
    pub fn tool_call(call: ToolCall) -> Self {
        Self {
            tool_calls: vec![call],
            finish_reason: FinishReason::ToolUse,
            ..Self::message("")
        }
    }
}

impl From<&str> for ProviderResponse {
    fn from(content: &str) -> Self {
        Self::message(content)
    }
}

impl From<String> for ProviderResponse {
    fn from(content: String) -> Self {
        Self::message(content)
    }
}

/// A boxed error from a transport or other lower layer.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// Why a provider could not produce a response.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// The request could not be encoded for this provider.
    #[error("Invalid provider request: {0}")]
    InvalidRequest(String),
    /// The request was not sent or its response could not be read.
    #[error("Provider transport failed")]
    Transport(#[source] BoxError),
    /// No response arrived within the configured timeout.
    #[error("Provider request timed out after {0:?}")]
    Timeout(Duration),
    /// The provider answered with an error status.
    #[error("Provider returned HTTP {status}: {message}")]
    Api {
        status: u16,
        /// The provider's error type, such as `rate_limit_error`.
        kind: Option<String>,
        message: String,
        request_id: Option<String>,
    },
    /// The provider's response could not be decoded.
    #[error("Invalid provider response: {0}")]
    InvalidResponse(String),
}

/// A source of model responses, shared across turns.
///
/// Implementations are `Send + Sync` and used through `Arc<dyn Provider>`.
pub trait Provider: Debug + Send + Sync {
    /// A stable provider name, such as `anthropic`, recorded with responses.
    fn name(&self) -> &str;

    /// Token limits for `model`, or for the provider's default model.
    fn model_limits(&self, model: Option<&str>) -> ModelLimits;

    /// Generate one response.
    ///
    /// Dropping the returned future cancels the request; a provider keeps no
    /// per-request state, so the next call is unaffected.
    fn generate(
        &self,
        request: ProviderRequest,
    ) -> BoxFuture<'_, std::result::Result<ProviderResponse, ProviderError>>;
}
