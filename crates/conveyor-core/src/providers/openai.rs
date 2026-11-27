//! The OpenAI Responses API adapter.
//!
//! Requests are encoded here and sent through any [`HttpTransport`]. Every
//! request sets `store: false`: Conveyor keeps the transcript, so each
//! request carries the whole conversation, and a reasoning model keeps its
//! reasoning between requests through the encrypted items saved as
//! [`ReplayState`]. The adapter applies its own timeout and retry policy, so
//! a Tokio runtime with timers enabled must drive [`Provider::generate`].

use std::{fmt, sync::Arc, time::Duration};

use conveyor_http::{Bytes, HeaderValue, HttpTransport, Method, Request, Response, Uri, header};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{
    FinishReason, ModelLimits, Provider, ProviderError, ProviderMessage, ProviderRequest,
    ProviderResponse, ReplayState, ToolDefinition,
    retry::{self, Failure, RetryPolicy},
};
use crate::{
    BoxFuture, Error,
    models::{JsonObject, ToolCall, string_enum},
};

/// The provider name used in responses and replay state.
pub const PROVIDER_NAME: &str = "openai";
pub const DEFAULT_MODEL: &str = "gpt-5.5";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: u32 = 1_050_000;
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 128_000;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_MAX_RETRIES: u32 = 2;

const DEFAULT_LIMITS: ModelLimits =
    match ModelLimits::new(DEFAULT_CONTEXT_WINDOW_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS) {
        Ok(limits) => limits,
        Err(_) => panic!("default OpenAI limits are valid"),
    };
/// Error codes and types that fail the same way however often they are
/// retried: an oversized request, and exhausted credit or spend limits.
const FINAL_ERRORS: [&str; 6] = [
    "context_length_exceeded",
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
];
/// Marks a failed tool result, since a function call output has no error
/// field of its own.
const FAILED_RESULT_PREFIX: &str = "Error: ";

string_enum! {
    /// How much a reasoning model thinks before it answers. Models accept
    /// different subsets of these levels.
    pub enum ReasoningEffort {
        None => "none", Minimal => "minimal", Low => "low", Medium => "medium",
        High => "high", XHigh => "xhigh", Max => "max"
    }
}

/// OpenAI adapter settings. Validated by [`OpenAiProvider::new`].
#[derive(Clone)]
pub struct OpenAiConfig {
    api_key: String,
    /// The API root, such as `https://api.openai.com/v1`; requests go to
    /// `{base_url}/responses`.
    pub base_url: String,
    /// Sent as `OpenAI-Organization` when set.
    pub organization: Option<String>,
    /// Sent as `OpenAI-Project` when set.
    pub project: Option<String>,
    /// The model used when a request names none.
    pub model: String,
    pub limits: ModelLimits,
    /// The reasoning effort to request; the model's default when unset.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The deadline for each attempt, including reading the body.
    pub timeout: Duration,
    /// Retries after the first attempt for connection failures, timeouts,
    /// and HTTP 408, 409, 429, or 5xx responses, except billing and
    /// context-length errors.
    pub max_retries: u32,
}

impl OpenAiConfig {
    /// Default settings, authenticated with `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            organization: None,
            project: None,
            model: DEFAULT_MODEL.into(),
            limits: DEFAULT_LIMITS,
            reasoning_effort: None,
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Read `OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL`,
    /// `OPENAI_ORG_ID`, and `OPENAI_PROJECT_ID`.
    ///
    /// Returns `None` when no non-empty API key is set.
    pub fn from_env() -> Option<Self> {
        let variable = |name| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
        let mut config = Self::new(variable("OPENAI_API_KEY")?);
        if let Some(base_url) = variable("OPENAI_BASE_URL") {
            config.base_url = base_url;
        }
        config.organization = variable("OPENAI_ORG_ID");
        config.project = variable("OPENAI_PROJECT_ID");
        Some(config)
    }
}

impl fmt::Debug for OpenAiConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("organization", &self.organization)
            .field("project", &self.project)
            .field("model", &self.model)
            .field("limits", &self.limits)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Generates responses with the OpenAI Responses API.
#[derive(Debug)]
pub struct OpenAiProvider {
    config: OpenAiConfig,
    endpoint: Uri,
    authorization: HeaderValue,
    organization: Option<HeaderValue>,
    project: Option<HeaderValue>,
    transport: Arc<dyn HttpTransport>,
}

impl OpenAiProvider {
    /// Validate `config` and send requests through `transport`.
    ///
    /// Fails when the API key, organization, or project cannot be sent as a
    /// header, or the base URL does not form a valid endpoint.
    pub fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> crate::Result<Self> {
        let mut authorization = HeaderValue::from_str(&format!("Bearer {}", config.api_key.trim()))
            .map_err(|_| Error::InvalidApiKey)?;
        authorization.set_sensitive(true);
        let header = |value: &Option<String>, name| {
            value
                .as_deref()
                .map(|value| {
                    HeaderValue::from_str(value).map_err(|_| Error::InvalidHeaderValue(name))
                })
                .transpose()
        };
        let organization = header(&config.organization, "OpenAI organization")?;
        let project = header(&config.project, "OpenAI project")?;
        let endpoint = retry::endpoint(&config.base_url, "/responses")?;
        Ok(Self {
            config,
            endpoint,
            authorization,
            organization,
            project,
            transport,
        })
    }

    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    async fn send(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let body = encode_request(&request, &self.config)?;
        let body = Bytes::from(
            serde_json::to_vec(&body)
                .map_err(|error| ProviderError::InvalidRequest(error.to_string()))?,
        );
        let policy = RetryPolicy {
            timeout: self.config.timeout,
            max_retries: self.config.max_retries,
        };
        let response = retry::send(
            self.transport.as_ref(),
            policy,
            || self.http_request(body.clone()),
            api_failure,
        )
        .await?;
        decode_response(response.body())
    }

    fn http_request(&self, body: Bytes) -> Result<Request<Bytes>, ProviderError> {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(self.endpoint.clone())
            .header(header::AUTHORIZATION, self.authorization.clone())
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json");
        if let Some(organization) = &self.organization {
            builder = builder.header("openai-organization", organization.clone());
        }
        if let Some(project) = &self.project {
            builder = builder.header("openai-project", project.clone());
        }
        builder
            .body(body)
            .map_err(|error| ProviderError::InvalidRequest(error.to_string()))
    }
}

impl Provider for OpenAiProvider {
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

fn encode_request(
    request: &ProviderRequest,
    config: &OpenAiConfig,
) -> Result<Value, ProviderError> {
    let (instructions, input) = encode_input(&request.messages);
    let mut body = Map::new();
    let model = request.model.as_deref().unwrap_or(&config.model);
    body.insert("model".into(), model.into());
    if let Some(instructions) = instructions {
        body.insert("instructions".into(), instructions.into());
    }
    body.insert("input".into(), Value::Array(input));
    // Conveyor keeps the transcript; without stored state, reasoning
    // continues only through its encrypted items.
    body.insert("store".into(), false.into());
    body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    let max_tokens = request
        .max_tokens
        .unwrap_or(config.limits.max_output_tokens());
    body.insert("max_output_tokens".into(), max_tokens.into());
    if let Some(temperature) = request.temperature {
        let temperature = serde_json::Number::from_f64(temperature).ok_or_else(|| {
            ProviderError::InvalidRequest("Temperature must be a finite number".into())
        })?;
        body.insert("temperature".into(), Value::Number(temperature));
    }
    if !request.tools.is_empty() {
        let tools = request.tools.iter().map(encode_tool).collect();
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(effort) = config.reasoning_effort {
        body.insert("reasoning".into(), json!({"effort": effort.as_str()}));
    }
    Ok(Value::Object(body))
}

/// The top-level instructions and the input items for `messages`.
///
/// Leading system messages become the instructions, unless they are the
/// whole conversation, which then stays in the input so it is not empty.
/// Later system messages are sent in place with the `developer` role.
fn encode_input(messages: &[ProviderMessage]) -> (Option<String>, Vec<Value>) {
    let leading = messages
        .iter()
        .take_while(|message| matches!(message, ProviderMessage::System { .. }))
        .count();
    let lifted = if leading < messages.len() { leading } else { 0 };
    let instructions: Vec<&str> = messages[..lifted]
        .iter()
        .map(ProviderMessage::content)
        .filter(|content| !content.trim().is_empty())
        .collect();

    let mut input = Vec::new();
    for message in &messages[lifted..] {
        match message {
            ProviderMessage::System { content } => {
                input.push(json!({"role": "developer", "content": content}));
            }
            ProviderMessage::User { content } => {
                input.push(json!({"role": "user", "content": content}));
            }
            ProviderMessage::Assistant {
                content,
                tool_calls,
                replay_state,
            } => encode_assistant(&mut input, content, tool_calls, replay_state.as_ref()),
            ProviderMessage::Tool {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let output = if *is_error {
                    format!("{FAILED_RESULT_PREFIX}{content}")
                } else {
                    content.clone()
                };
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": output,
                }));
            }
        }
    }
    if input.is_empty() {
        input.push(json!({"role": "user", "content": ""}));
    }
    let instructions = (!instructions.is_empty()).then(|| instructions.join("\n\n"));
    (instructions, input)
}

/// An assistant turn: the response's own output items when this provider
/// wrote it, otherwise its text and tool calls.
fn encode_assistant(
    input: &mut Vec<Value>,
    content: &str,
    tool_calls: &[ToolCall],
    replay_state: Option<&ReplayState>,
) {
    if let Some(state) =
        replay_state.filter(|state| state.provider == PROVIDER_NAME && !state.items.is_empty())
    {
        replay_items(input, &state.items);
        return;
    }
    if !content.is_empty() {
        input.push(json!({"role": "assistant", "content": content}));
    }
    for call in tool_calls {
        input.push(function_call_item(call));
    }
}

fn function_call_item(call: &ToolCall) -> Value {
    json!({
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": Value::Object(call.arguments.clone()).to_string(),
    })
}

/// Send saved output items back unchanged and in their original order.
///
/// A reasoning item without encrypted content cannot be resolved when
/// nothing is stored, so it is dropped, and the item after it loses its id
/// so that nothing refers to the missing reasoning: a message becomes plain
/// assistant text, and other items keep everything else.
fn replay_items(input: &mut Vec<Value>, items: &[JsonObject]) {
    let mut unpaired = false;
    for item in items {
        if item_type(item) == Some("reasoning") {
            let resolvable = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(|ciphertext| !ciphertext.is_empty());
            if resolvable {
                input.push(Value::Object(item.clone()));
            } else {
                unpaired = true;
            }
            continue;
        }
        if !std::mem::take(&mut unpaired) {
            input.push(Value::Object(item.clone()));
        } else if item_type(item) == Some("message") {
            let mut refused = false;
            let text = message_text(item, &mut refused);
            if !text.is_empty() {
                input.push(json!({"role": "assistant", "content": text}));
            }
        } else {
            let mut item = item.clone();
            item.remove("id");
            input.push(Value::Object(item));
        }
    }
}

fn item_type(item: &JsonObject) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

/// A function tool. Strict schema validation is off: Conveyor's schemas
/// have optional fields, which strict mode does not allow.
fn encode_tool(tool: &ToolDefinition) -> Value {
    let parameters = if tool.parameters.is_empty() {
        json!({"type": "object", "properties": {}})
    } else {
        Value::Object(tool.parameters.clone())
    };
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
        "strict": false,
    })
}

#[derive(Deserialize)]
struct WireResponse {
    id: String,
    model: String,
    status: Option<String>,
    incomplete_details: Option<WireIncomplete>,
    error: Option<WireError>,
    #[serde(default)]
    output: Vec<JsonObject>,
    #[serde(default)]
    usage: Value,
}

#[derive(Deserialize)]
struct WireIncomplete {
    reason: Option<String>,
}

#[derive(Deserialize)]
struct WireError {
    code: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

fn decode_response(body: &[u8]) -> Result<ProviderResponse, ProviderError> {
    let response: WireResponse = serde_json::from_slice(body)
        .map_err(|error| ProviderError::InvalidResponse(error.to_string()))?;
    // `None` when the response completed, otherwise why it stopped early.
    let incomplete = match response.status.as_deref() {
        None | Some("completed") => None,
        Some("incomplete") => Some(
            response
                .incomplete_details
                .and_then(|details| details.reason),
        ),
        Some(status @ ("failed" | "cancelled")) => {
            let mut message = format!("Response {status}");
            if let Some(error) = response.error {
                for part in [error.code, error.message].into_iter().flatten() {
                    message.push_str(": ");
                    message.push_str(&part);
                }
            }
            return Err(ProviderError::InvalidResponse(message));
        }
        Some(status) => {
            return Err(ProviderError::InvalidResponse(format!(
                "Response ended with status {status}"
            )));
        }
    };

    let mut text = Vec::new();
    let mut tool_calls = Vec::new();
    let mut refused = false;
    for item in &response.output {
        match item_type(item) {
            Some("message") => {
                let part = message_text(item, &mut refused);
                if !part.is_empty() {
                    text.push(part);
                }
            }
            Some("function_call") => match function_call(item) {
                Ok(call) => tool_calls.push(call),
                // A call cut short by the output limit is dropped; the
                // finish reason already says the response is incomplete.
                Err(_) if incomplete.is_some() => {}
                Err(error) => return Err(ProviderError::InvalidResponse(error)),
            },
            _ => {}
        }
    }

    let finish_reason = match incomplete.as_ref().map(Option::as_deref) {
        None if !tool_calls.is_empty() => FinishReason::ToolUse,
        None if refused => FinishReason::Refusal,
        None => FinishReason::Stop,
        Some(Some("max_output_tokens")) => FinishReason::MaxTokens,
        Some(Some("content_filter")) => FinishReason::Refusal,
        Some(_) => FinishReason::Unknown,
    };
    let mut raw = JsonObject::new();
    raw.insert("provider".into(), PROVIDER_NAME.into());
    raw.insert("model".into(), response.model.into());
    raw.insert("response_id".into(), response.id.into());
    raw.insert("usage".into(), retry::without_nulls(response.usage));
    Ok(ProviderResponse {
        content: text.join("\n").trim().to_owned(),
        tool_calls,
        finish_reason,
        raw,
        replay_state: (!response.output.is_empty()).then(|| ReplayState {
            provider: PROVIDER_NAME.into(),
            items: response.output,
        }),
    })
}

/// A message item's text: its `output_text` parts, and any `refusal` parts,
/// which also set `refused`.
fn message_text(item: &JsonObject, refused: &mut bool) -> String {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    let mut text = String::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => text.push_str(part["text"].as_str().unwrap_or_default()),
            Some("refusal") => {
                *refused = true;
                text.push_str(part["refusal"].as_str().unwrap_or_default());
            }
            _ => {}
        }
    }
    text
}

/// A function call item as a tool call identified by its `call_id`, which
/// its result must name.
fn function_call(item: &JsonObject) -> Result<ToolCall, String> {
    let field = |key| {
        item.get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    };
    let name = field("name").ok_or("Function call has no name")?;
    let id = field("call_id").ok_or_else(|| format!("Function call {name} has no call_id"))?;
    let arguments = match item.get("arguments") {
        None | Some(Value::Null) => JsonObject::new(),
        Some(Value::String(arguments)) if arguments.trim().is_empty() => JsonObject::new(),
        Some(Value::String(arguments)) => match serde_json::from_str(arguments) {
            Ok(Value::Object(arguments)) => arguments,
            Ok(_) => return Err(format!("Function call {name} arguments are not an object")),
            Err(error) => {
                return Err(format!(
                    "Function call {name} has invalid arguments: {error}"
                ));
            }
        },
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(_) => return Err(format!("Function call {name} arguments are not an object")),
    };
    Ok(ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

fn api_failure(response: &Response<Bytes>) -> Failure {
    #[derive(Deserialize)]
    struct WireErrorBody {
        error: WireError,
    }

    let status = response.status();
    let headers = response.headers();
    let error = serde_json::from_slice::<WireErrorBody>(response.body())
        .map(|body| body.error)
        .ok();
    let (code, kind, message) = match error {
        Some(error) => (error.code, error.kind, error.message),
        None => (None, None, None),
    };
    let message = message.unwrap_or_else(|| retry::error_text(response));
    let final_error = [&code, &kind]
        .into_iter()
        .flatten()
        .any(|value| FINAL_ERRORS.contains(&value.as_str()));
    Failure {
        retryable: !final_error && retry::should_retry(status, headers),
        retry_after: retry::retry_after(headers).or_else(|| retry::hinted_retry(&message)),
        error: ProviderError::Api {
            status: status.as_u16(),
            // The code is the most specific name, such as
            // `context_length_exceeded`; the type is broader.
            kind: code.or(kind),
            message,
            request_id: retry::header_str(headers, "x-request-id").map(str::to_owned),
        },
    }
}
