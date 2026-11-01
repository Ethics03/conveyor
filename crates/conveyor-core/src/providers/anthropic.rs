//! The Anthropic Messages API adapter.
//!
//! Requests are encoded here and sent through any [`HttpTransport`]. The
//! adapter applies its own timeout and retry policy, so a Tokio runtime with
//! timers enabled must drive [`Provider::generate`].

use std::{fmt, sync::Arc, time::Duration};

use conveyor_http::{
    Bytes, HeaderMap, HeaderValue, HttpTransport, Method, Request, Response, StatusCode, Uri,
    header,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
    BoxError, FinishReason, ModelLimits, Provider, ProviderError, ProviderMessage, ProviderRequest,
    ProviderResponse, ReplayState, ToolDefinition,
};
use crate::{
    BoxFuture, Error,
    models::{JsonObject, ToolCall},
};

/// The provider name recorded with responses and replay state.
pub const PROVIDER_NAME: &str = "anthropic";
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const API_VERSION: &str = "2023-06-01";
pub const COMPACTION_BETA: &str = "compact-2026-01-12";
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: u32 = 200_000;
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4_096;
pub const DEFAULT_COMPACTION_TRIGGER_TOKENS: u32 = 150_000;
/// The smallest native compaction trigger the API accepts.
pub const MIN_COMPACTION_TRIGGER_TOKENS: u32 = 50_000;
/// How far below the local planner's trigger native compaction fires.
pub const COMPACTION_LOCAL_FALLBACK_MARGIN_TOKENS: u32 = 8_192;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_MAX_RETRIES: u32 = 2;

const DEFAULT_LIMITS: ModelLimits =
    match ModelLimits::new(DEFAULT_CONTEXT_WINDOW_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS) {
        Ok(limits) => limits,
        Err(_) => panic!("default Anthropic limits are valid"),
    };
const COMPACTION_MODEL_MARKERS: [&str; 8] = [
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
];
const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(500);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(8);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const ERROR_BODY_LIMIT: usize = 1_000;

/// Anthropic adapter settings. Validated by [`AnthropicProvider::new`].
#[derive(Clone)]
pub struct AnthropicConfig {
    api_key: String,
    /// The API origin, without the `/v1/messages` path.
    pub base_url: String,
    /// The model used when a request names none.
    pub model: String,
    pub limits: ModelLimits,
    /// Request automatic prompt caching for each call.
    pub prompt_caching: bool,
    /// Ask supported models to compact context server-side.
    pub native_compaction: bool,
    /// The highest native compaction trigger, in input tokens.
    pub compaction_trigger_tokens: u32,
    /// The deadline for each attempt, including reading the body.
    pub timeout: Duration,
    /// Retries after the first attempt for connection failures, timeouts,
    /// and HTTP 408, 409, 429, or 5xx responses.
    pub max_retries: u32,
}

impl AnthropicConfig {
    /// Defaults matching the Python adapter, authenticated with `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            limits: DEFAULT_LIMITS,
            prompt_caching: true,
            native_compaction: true,
            compaction_trigger_tokens: DEFAULT_COMPACTION_TRIGGER_TOKENS,
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Read `ANTHROPIC_API_KEY` and an optional `ANTHROPIC_BASE_URL`.
    ///
    /// Returns `None` when no non-empty API key is set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|key| !key.trim().is_empty())?;
        let mut config = Self::new(api_key);
        if let Some(base_url) = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .filter(|url| !url.trim().is_empty())
        {
            config.base_url = base_url;
        }
        Some(config)
    }
}

impl fmt::Debug for AnthropicConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("limits", &self.limits)
            .field("prompt_caching", &self.prompt_caching)
            .field("native_compaction", &self.native_compaction)
            .field("compaction_trigger_tokens", &self.compaction_trigger_tokens)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Generates responses with the Anthropic Messages API.
#[derive(Debug)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    endpoint: Uri,
    api_key: HeaderValue,
    transport: Arc<dyn HttpTransport>,
}

impl AnthropicProvider {
    /// Validate `config` and send requests through `transport`.
    ///
    /// Fails when the compaction trigger is below
    /// [`MIN_COMPACTION_TRIGGER_TOKENS`], the API key cannot be sent as a
    /// header, or the base URL does not form a valid endpoint.
    pub fn new(config: AnthropicConfig, transport: Arc<dyn HttpTransport>) -> crate::Result<Self> {
        if config.compaction_trigger_tokens < MIN_COMPACTION_TRIGGER_TOKENS {
            return Err(Error::CompactionTriggerTooLow {
                minimum: MIN_COMPACTION_TRIGGER_TOKENS,
            });
        }
        let mut api_key =
            HeaderValue::from_str(&config.api_key).map_err(|_| Error::InvalidApiKey)?;
        api_key.set_sensitive(true);
        let endpoint = format!("{}/v1/messages", config.base_url.trim_end_matches('/'))
            .parse::<Uri>()
            .ok()
            .filter(|uri| uri.scheme().is_some() && uri.authority().is_some())
            .ok_or(Error::InvalidBaseUrl)?;
        Ok(Self {
            config,
            endpoint,
            api_key,
            transport,
        })
    }

    pub fn config(&self) -> &AnthropicConfig {
        &self.config
    }

    async fn send(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let encoded = encode_request(&request, &self.config)?;
        let body = Bytes::from(
            serde_json::to_vec(&encoded.body)
                .map_err(|error| ProviderError::InvalidRequest(error.to_string()))?,
        );
        let mut attempt = 0;
        loop {
            let http_request = self.http_request(body.clone(), encoded.compaction)?;
            let outcome =
                tokio::time::timeout(self.config.timeout, self.transport.send(http_request)).await;
            let failure = match outcome {
                Ok(Ok(response)) if response.status().is_success() => {
                    return decode_response(response.body());
                }
                Ok(Ok(response)) => api_failure(&response),
                Ok(Err(error)) => Failure {
                    retryable: error.is_retryable(),
                    retry_after: None,
                    error: ProviderError::Transport(BoxError::from(error)),
                },
                Err(_) => Failure {
                    retryable: true,
                    retry_after: None,
                    error: ProviderError::Timeout(self.config.timeout),
                },
            };
            if !failure.retryable || attempt >= self.config.max_retries {
                return Err(failure.error);
            }
            tokio::time::sleep(failure.retry_after.unwrap_or_else(|| backoff(attempt))).await;
            attempt += 1;
        }
    }

    fn http_request(&self, body: Bytes, compaction: bool) -> Result<Request<Bytes>, ProviderError> {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(self.endpoint.clone())
            .header("x-api-key", self.api_key.clone())
            .header("anthropic-version", API_VERSION)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json");
        if compaction {
            builder = builder.header("anthropic-beta", COMPACTION_BETA);
        }
        builder
            .body(body)
            .map_err(|error| ProviderError::InvalidRequest(error.to_string()))
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model_limits(&self, _model: Option<&str>) -> ModelLimits {
        self.config.limits
    }

    fn generate(
        &self,
        request: ProviderRequest,
    ) -> BoxFuture<'_, Result<ProviderResponse, ProviderError>> {
        Box::pin(self.send(request))
    }
}

struct EncodedRequest {
    body: Value,
    compaction: bool,
}

fn encode_request(
    request: &ProviderRequest,
    config: &AnthropicConfig,
) -> Result<EncodedRequest, ProviderError> {
    let model = request.model.as_deref().unwrap_or(&config.model);
    let mut body = Map::new();
    body.insert("model".into(), model.into());
    let max_tokens = request
        .max_tokens
        .unwrap_or(config.limits.max_output_tokens());
    body.insert("max_tokens".into(), max_tokens.into());
    body.insert("messages".into(), encode_messages(&request.messages));
    if let Some(temperature) = request.temperature {
        let temperature = serde_json::Number::from_f64(temperature).ok_or_else(|| {
            ProviderError::InvalidRequest("Temperature must be a finite number".into())
        })?;
        body.insert("temperature".into(), Value::Number(temperature));
    }
    let system = system_instructions(&request.messages);
    if !system.is_empty() {
        body.insert("system".into(), system.into());
    }
    if !request.tools.is_empty() {
        let tools = request.tools.iter().map(encode_tool).collect();
        body.insert("tools".into(), Value::Array(tools));
    }
    if config.prompt_caching {
        body.insert("cache_control".into(), json!({"type": "ephemeral"}));
    }
    let compaction = compaction_trigger(request, config, model);
    if let Some(trigger) = compaction {
        body.insert(
            "context_management".into(),
            json!({"edits": [{
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": trigger},
            }]}),
        );
    }
    Ok(EncodedRequest {
        body: Value::Object(body),
        compaction: compaction.is_some(),
    })
}

fn system_instructions(messages: &[ProviderMessage]) -> String {
    let instructions: Vec<_> = messages
        .iter()
        .filter_map(|message| match message {
            ProviderMessage::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    instructions.join("\n\n")
}

/// Convert messages, grouping consecutive tool results into one user turn.
fn encode_messages(messages: &[ProviderMessage]) -> Value {
    let mut converted = Vec::new();
    let mut tool_results = Vec::new();
    for message in messages {
        let encoded = match message {
            ProviderMessage::System { .. } => continue,
            ProviderMessage::Tool {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                });
                if *is_error {
                    block["is_error"] = Value::Bool(true);
                }
                tool_results.push(block);
                continue;
            }
            ProviderMessage::User { content } => json!({"role": "user", "content": content}),
            ProviderMessage::Assistant {
                content,
                tool_calls,
                replay_state,
            } => encode_assistant(content, tool_calls, replay_state.as_ref()),
        };
        if !tool_results.is_empty() {
            let results = std::mem::take(&mut tool_results);
            converted.push(json!({"role": "user", "content": results}));
        }
        converted.push(encoded);
    }
    if !tool_results.is_empty() {
        converted.push(json!({"role": "user", "content": tool_results}));
    }
    if converted.is_empty() {
        converted.push(json!({"role": "user", "content": ""}));
    }
    Value::Array(converted)
}

fn encode_assistant(
    content: &str,
    tool_calls: &[ToolCall],
    replay_state: Option<&ReplayState>,
) -> Value {
    let mut blocks = replay_blocks(replay_state);
    if tool_calls.is_empty() && blocks.is_empty() {
        return json!({"role": "assistant", "content": content});
    }
    if !content.is_empty() {
        blocks.push(json!({"type": "text", "text": content}));
    }
    blocks.extend(tool_calls.iter().map(|call| {
        json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        })
    }));
    json!({"role": "assistant", "content": blocks})
}

/// Replay this provider's well-formed compaction items; skip everything else.
fn replay_blocks(state: Option<&ReplayState>) -> Vec<Value> {
    let Some(state) = state.filter(|state| state.provider == PROVIDER_NAME) else {
        return Vec::new();
    };
    state
        .items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("compaction"))
        .filter_map(|item| {
            let content = optional_string(item, "content")?;
            let encrypted = optional_string(item, "encrypted_content")?;
            let mut block = json!({"type": "compaction", "content": content});
            if let Some(encrypted) = encrypted {
                block["encrypted_content"] = encrypted.into();
            }
            Some(block)
        })
        .collect()
}

/// `Some(None)` for a missing or null field, `None` for a non-string value.
fn optional_string<'a>(item: &'a JsonObject, key: &str) -> Option<Option<&'a str>> {
    match item.get(key) {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(value)) => Some(Some(value)),
        Some(_) => None,
    }
}

fn encode_tool(tool: &ToolDefinition) -> Value {
    let input_schema = if tool.parameters.is_empty() {
        json!({"type": "object", "properties": {}})
    } else {
        Value::Object(tool.parameters.clone())
    };
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema,
    })
}

/// The native compaction trigger, set below the local fallback's trigger.
fn compaction_trigger(
    request: &ProviderRequest,
    config: &AnthropicConfig,
    model: &str,
) -> Option<u32> {
    if !config.native_compaction || !supports_native_compaction(model) {
        return None;
    }
    let local = request.compaction_trigger_tokens?;
    let trigger = config
        .compaction_trigger_tokens
        .min(local.saturating_sub(COMPACTION_LOCAL_FALLBACK_MARGIN_TOKENS));
    (trigger >= MIN_COMPACTION_TRIGGER_TOKENS).then_some(trigger)
}

fn supports_native_compaction(model: &str) -> bool {
    let model = model.to_lowercase();
    COMPACTION_MODEL_MARKERS
        .iter()
        .any(|marker| model.contains(marker))
}

#[derive(Deserialize)]
struct WireResponse {
    id: String,
    model: String,
    content: Vec<WireBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Value,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonObject,
    },
    Compaction {
        content: Option<String>,
        encrypted_content: Option<String>,
    },
    #[serde(other)]
    Other,
}

fn decode_response(body: &[u8]) -> Result<ProviderResponse, ProviderError> {
    let response: WireResponse = serde_json::from_slice(body)
        .map_err(|error| ProviderError::InvalidResponse(error.to_string()))?;
    let mut text = Vec::new();
    let mut tool_calls = Vec::new();
    let mut compaction = Vec::new();
    for block in response.content {
        match block {
            WireBlock::Text { text: part } => text.push(part),
            WireBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id,
                name,
                arguments: input,
            }),
            WireBlock::Compaction {
                content,
                encrypted_content,
            } => {
                let mut item = JsonObject::new();
                item.insert("type".into(), "compaction".into());
                item.insert("content".into(), content.map_or(Value::Null, Value::String));
                if let Some(encrypted) = encrypted_content {
                    item.insert("encrypted_content".into(), encrypted.into());
                }
                compaction.push(item);
            }
            WireBlock::Other => {}
        }
    }
    let mut raw = JsonObject::new();
    raw.insert("provider".into(), PROVIDER_NAME.into());
    raw.insert("model".into(), response.model.into());
    raw.insert("response_id".into(), response.id.into());
    raw.insert("usage".into(), without_nulls(response.usage));
    Ok(ProviderResponse {
        content: text.join("\n").trim().to_owned(),
        tool_calls,
        finish_reason: finish_reason(response.stop_reason.as_deref()),
        raw,
        replay_state: (!compaction.is_empty()).then(|| ReplayState {
            provider: PROVIDER_NAME.into(),
            items: compaction,
        }),
    })
}

fn finish_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolUse,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::MaxTokens,
        Some("refusal") => FinishReason::Refusal,
        Some("pause_turn") => FinishReason::Pause,
        _ => FinishReason::Unknown,
    }
}

/// Drop null object fields recursively, matching the SDK's usage dump.
fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key, without_nulls(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

struct Failure {
    error: ProviderError,
    retryable: bool,
    retry_after: Option<Duration>,
}

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireError,
}

#[derive(Deserialize)]
struct WireError {
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

fn api_failure(response: &Response<Bytes>) -> Failure {
    let status = response.status();
    let headers = response.headers();
    let (kind, message) = match serde_json::from_slice::<WireErrorBody>(response.body()) {
        Ok(WireErrorBody { error }) => (error.kind, error.message),
        Err(_) => (None, None),
    };
    let message = message.unwrap_or_else(|| {
        let body = String::from_utf8_lossy(response.body());
        let body = body.trim();
        match body.char_indices().nth(ERROR_BODY_LIMIT) {
            Some((end, _)) => format!("{}…", &body[..end]),
            None if body.is_empty() => status.canonical_reason().unwrap_or("").to_owned(),
            None => body.to_owned(),
        }
    });
    Failure {
        retryable: should_retry(status, headers),
        retry_after: retry_after(headers),
        error: ProviderError::Api {
            status: status.as_u16(),
            kind,
            message,
            request_id: header_str(headers, "request-id").map(str::to_owned),
        },
    }
}

/// Honor `x-should-retry`, otherwise retry timeouts, conflicts, rate limits,
/// and server errors.
fn should_retry(status: StatusCode, headers: &HeaderMap) -> bool {
    match header_str(headers, "x-should-retry") {
        Some("true") => true,
        Some("false") => false,
        _ => matches!(status.as_u16(), 408 | 409 | 429) || status.is_server_error(),
    }
}

/// A server-requested delay of at most a minute, in milliseconds or seconds.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let millis = header_str(headers, "retry-after-ms")
        .and_then(|value| value.parse::<f64>().ok())
        .map(|millis| millis / 1_000.0);
    let seconds = millis.or_else(|| {
        header_str(headers, "retry-after").and_then(|value| value.parse::<f64>().ok())
    })?;
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|delay| !delay.is_zero() && *delay <= MAX_RETRY_AFTER)
}

/// Exponential backoff from 0.5 seconds, capped at 8 seconds.
fn backoff(attempt: u32) -> Duration {
    INITIAL_RETRY_DELAY
        .saturating_mul(2u32.saturating_pow(attempt))
        .min(MAX_RETRY_DELAY)
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}
