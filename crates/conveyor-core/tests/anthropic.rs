use std::{
    collections::VecDeque,
    future,
    sync::{Arc, Mutex},
    time::Duration,
};

use conveyor_core::{
    Error,
    models::{JsonObject, ToolCall},
    providers::{
        FinishReason, Provider, ProviderError, ProviderMessage, ProviderRequest, ReplayState,
        ToolDefinition,
        anthropic::{AnthropicConfig, AnthropicProvider, COMPACTION_BETA},
    },
};
use conveyor_http::{
    BoxFuture, Bytes, Error as HttpError, ErrorKind, HttpTransport, Request, Response,
};
use serde_json::{Value, json};
use tokio::time::Instant;

enum Step {
    Reply(Response<Bytes>),
    Fail(ErrorKind),
    Hang,
}

#[derive(Debug, Default)]
struct ScriptedTransport {
    steps: Mutex<VecDeque<Step>>,
    requests: Mutex<Vec<Request<Bytes>>>,
}

impl std::fmt::Debug for Step {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Step")
    }
}

impl ScriptedTransport {
    fn new(steps: impl IntoIterator<Item = Step>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into_iter().collect()),
            requests: Mutex::default(),
        })
    }

    fn requests(&self) -> Vec<Request<Bytes>> {
        self.requests.lock().unwrap().clone()
    }

    fn bodies(&self) -> Vec<Value> {
        self.requests()
            .iter()
            .map(|request| serde_json::from_slice(request.body()).unwrap())
            .collect()
    }
}

impl HttpTransport for ScriptedTransport {
    fn send(&self, request: Request<Bytes>) -> BoxFuture<'_, Result<Response<Bytes>, HttpError>> {
        self.requests.lock().unwrap().push(request);
        let step = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted step");
        match step {
            Step::Reply(response) => Box::pin(future::ready(Ok(response))),
            Step::Fail(kind) => Box::pin(future::ready(Err(HttpError::new(kind, "scripted")))),
            Step::Hang => Box::pin(future::pending()),
        }
    }
}

fn reply(status: u16, body: Value) -> Step {
    reply_with(status, &[], body)
}

fn reply_with(status: u16, headers: &[(&str, &str)], body: Value) -> Step {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    Step::Reply(builder.body(Bytes::from(body.to_string())).unwrap())
}

fn success(content: Value, stop_reason: &str) -> Step {
    reply(
        200,
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 12, "output_tokens": 3, "cache_creation": null},
        }),
    )
}

fn provider(transport: &Arc<ScriptedTransport>) -> AnthropicProvider {
    provider_with(transport, |_| {})
}

fn provider_with(
    transport: &Arc<ScriptedTransport>,
    configure: impl FnOnce(&mut AnthropicConfig),
) -> AnthropicProvider {
    let mut config = AnthropicConfig::new("sk-test");
    configure(&mut config);
    AnthropicProvider::new(config, transport.clone()).unwrap()
}

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

fn user(content: &str) -> ProviderRequest {
    ProviderRequest::new(vec![ProviderMessage::user(content)])
}

#[tokio::test]
async fn requests_encode_messages_tools_and_headers() {
    let transport = ScriptedTransport::new([success(json!([]), "end_turn")]);
    let provider = provider(&transport);
    let call = |id: &str| ToolCall {
        id: id.into(),
        name: "read_file".into(),
        arguments: object(json!({"path": "a.txt"})),
    };
    let mut request = ProviderRequest::new(vec![
        ProviderMessage::system("Be brief"),
        ProviderMessage::system("Temporal context"),
        ProviderMessage::user("read two files"),
        ProviderMessage::Assistant {
            content: "Reading".into(),
            tool_calls: vec![call("call_1"), call("call_2")],
            replay_state: None,
        },
        ProviderMessage::tool_result("call_1", "read_file", "one", false),
        ProviderMessage::tool_result("call_2", "read_file", "denied", true),
        ProviderMessage::assistant("done"),
    ]);
    request.tools = vec![
        ToolDefinition::new("read_file", "Read", object(json!({"type": "object"}))),
        ToolDefinition::new("noop", "Nothing", JsonObject::new()),
    ];
    provider.generate(request).await.unwrap();

    let sent = transport.requests();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method(), "POST");
    assert_eq!(sent[0].uri(), "https://api.anthropic.com/v1/messages");
    let headers = sent[0].headers();
    assert_eq!(headers["x-api-key"], "sk-test");
    assert!(headers["x-api-key"].is_sensitive());
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert_eq!(headers["content-type"], "application/json");
    assert!(!headers.contains_key("anthropic-beta"));

    assert_eq!(
        transport.bodies()[0],
        json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": "Be brief\n\nTemporal context",
            "cache_control": {"type": "ephemeral"},
            "messages": [
                {"role": "user", "content": "read two files"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Reading"},
                    {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.txt"}},
                    {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "a.txt"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "one"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "denied", "is_error": true},
                ]},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [
                {"name": "read_file", "description": "Read", "input_schema": {"type": "object"}},
                {"name": "noop", "description": "Nothing",
                 "input_schema": {"type": "object", "properties": {}}},
            ],
        })
    );
}

#[tokio::test]
async fn request_options_override_defaults_and_trailing_tool_results_are_flushed() {
    let transport = ScriptedTransport::new([success(json!([]), "end_turn")]);
    let provider = provider_with(&transport, |config| {
        config.prompt_caching = false;
        config.base_url = "http://localhost:8080/".into();
    });
    let mut request = ProviderRequest::new(vec![ProviderMessage::tool_result(
        "call_1", "bash", "ok", false,
    )]);
    request.model = Some("claude-haiku-4-5".into());
    request.temperature = Some(0.5);
    request.max_tokens = Some(256);
    provider.generate(request).await.unwrap();

    assert_eq!(
        transport.requests()[0].uri(),
        "http://localhost:8080/v1/messages"
    );
    assert_eq!(
        transport.bodies()[0],
        json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ]}],
        })
    );
}

#[tokio::test]
async fn an_empty_conversation_sends_one_empty_user_message() {
    let transport = ScriptedTransport::new([success(json!([]), "end_turn")]);
    provider(&transport)
        .generate(ProviderRequest::new(vec![ProviderMessage::system("Only")]))
        .await
        .unwrap();
    assert_eq!(
        transport.bodies()[0]["messages"],
        json!([{"role": "user", "content": ""}])
    );
}

#[tokio::test]
async fn native_compaction_fires_below_the_local_trigger_on_supported_models() {
    let transport = ScriptedTransport::new((0..4).map(|_| success(json!([]), "end_turn")));
    let provider = provider(&transport);
    let with_trigger = |model: &str, trigger: u32| {
        let mut request = user("hi");
        request.model = Some(model.into());
        request.compaction_trigger_tokens = Some(trigger);
        request
    };
    for request in [
        with_trigger("claude-sonnet-4-6", 160_000),
        with_trigger("claude-opus-5-5", 100_000),
        with_trigger("claude-sonnet-4-6", 58_000),
        with_trigger("claude-haiku-4-5", 160_000),
    ] {
        provider.generate(request).await.unwrap();
    }

    let requests = transport.requests();
    let bodies = transport.bodies();
    let edit = |value: u32| {
        json!({"edits": [{"type": "compact_20260112",
                          "trigger": {"type": "input_tokens", "value": value}}]})
    };
    assert_eq!(bodies[0]["context_management"], edit(150_000));
    assert_eq!(bodies[1]["context_management"], edit(91_808));
    for request in &requests[..2] {
        assert_eq!(request.headers()["anthropic-beta"], COMPACTION_BETA);
    }
    for (request, body) in requests[2..].iter().zip(&bodies[2..]) {
        assert!(body.get("context_management").is_none());
        assert!(!request.headers().contains_key("anthropic-beta"));
    }
}

#[tokio::test]
async fn compaction_is_skipped_without_a_local_trigger_or_when_disabled() {
    let transport = ScriptedTransport::new((0..2).map(|_| success(json!([]), "end_turn")));
    let disabled = provider_with(&transport, |config| config.native_compaction = false);
    let mut request = user("hi");
    request.compaction_trigger_tokens = Some(160_000);
    disabled.generate(request).await.unwrap();
    provider(&transport).generate(user("hi")).await.unwrap();
    assert!(
        transport
            .bodies()
            .iter()
            .all(|body| body.get("context_management").is_none())
    );
}

#[tokio::test]
async fn only_well_formed_anthropic_compaction_items_are_replayed() {
    let transport = ScriptedTransport::new([success(json!([]), "end_turn")]);
    let replayed = ReplayState::from_metadata(&json!({
        "provider": "anthropic",
        "items": [
            {"type": "compaction", "content": "summary", "encrypted_content": "opaque"},
            {"type": "compaction"},
            {"type": "compaction", "content": 7},
            {"type": "thinking", "content": "skip"},
        ],
    }))
    .unwrap();
    let foreign = ReplayState {
        provider: "other".into(),
        items: vec![object(json!({"type": "compaction", "content": "x"}))],
    };
    let request = ProviderRequest::new(vec![
        ProviderMessage::Assistant {
            content: String::new(),
            tool_calls: Vec::new(),
            replay_state: Some(replayed),
        },
        ProviderMessage::Assistant {
            content: "foreign".into(),
            tool_calls: Vec::new(),
            replay_state: Some(foreign),
        },
    ]);
    provider(&transport).generate(request).await.unwrap();
    assert_eq!(
        transport.bodies()[0]["messages"],
        json!([
            {"role": "assistant", "content": [
                {"type": "compaction", "content": "summary", "encrypted_content": "opaque"},
                {"type": "compaction", "content": null},
            ]},
            {"role": "assistant", "content": "foreign"},
        ])
    );
}

#[tokio::test]
async fn responses_decode_text_tool_calls_usage_and_compaction_state() {
    let transport = ScriptedTransport::new([success(
        json!([
            {"type": "compaction", "content": "summary", "encrypted_content": null},
            {"type": "thinking", "thinking": "hidden", "signature": "sig"},
            {"type": "text", "text": "  Let me check. "},
            {"type": "text", "text": "One moment.\n"},
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
        ]),
        "tool_use",
    )]);
    let response = provider(&transport).generate(user("hi")).await.unwrap();

    assert_eq!(response.content, "Let me check. \nOne moment.");
    assert_eq!(response.finish_reason, FinishReason::ToolUse);
    assert_eq!(response.tool_calls[0].id, "toolu_1");
    assert_eq!(response.tool_calls[0].arguments["command"], "ls");
    assert_eq!(
        Value::Object(response.raw),
        json!({
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "response_id": "msg_1",
            "usage": {"input_tokens": 12, "output_tokens": 3},
        })
    );
    let state = response.replay_state.unwrap();
    assert_eq!(
        state.to_metadata(),
        json!({"provider": "anthropic", "items": [{"type": "compaction", "content": "summary"}]})
    );
}

#[tokio::test]
async fn stop_reasons_map_to_neutral_finish_reasons() {
    let cases = [
        ("end_turn", FinishReason::Stop),
        ("stop_sequence", FinishReason::Stop),
        ("tool_use", FinishReason::ToolUse),
        ("max_tokens", FinishReason::MaxTokens),
        ("model_context_window_exceeded", FinishReason::MaxTokens),
        ("refusal", FinishReason::Refusal),
        ("pause_turn", FinishReason::Pause),
        ("something_new", FinishReason::Unknown),
    ];
    let transport =
        ScriptedTransport::new(cases.iter().map(|(reason, _)| success(json!([]), reason)));
    let provider = provider(&transport);
    for (reason, expected) in cases {
        let response = provider.generate(user("hi")).await.unwrap();
        assert_eq!(response.finish_reason, expected, "{reason}");
    }
}

#[tokio::test]
async fn malformed_success_bodies_are_invalid_responses() {
    let transport = ScriptedTransport::new([reply(200, json!({"id": "msg_1"}))]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert!(matches!(error, ProviderError::InvalidResponse(_)));
}

#[tokio::test]
async fn client_errors_are_reported_without_retrying() {
    let transport = ScriptedTransport::new([reply_with(
        400,
        &[("request-id", "req_1")],
        json!({"type": "error",
               "error": {"type": "invalid_request_error", "message": "bad tool schema"}}),
    )]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert_eq!(transport.requests().len(), 1);
    match error {
        ProviderError::Api {
            status,
            kind,
            message,
            request_id,
        } => {
            assert_eq!(status, 400);
            assert_eq!(kind.as_deref(), Some("invalid_request_error"));
            assert_eq!(message, "bad tool schema");
            assert_eq!(request_id.as_deref(), Some("req_1"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn overloads_and_connection_failures_retry_with_exponential_backoff() {
    let transport = ScriptedTransport::new([
        reply(
            529,
            json!({"error": {"type": "overloaded_error", "message": "busy"}}),
        ),
        Step::Fail(ErrorKind::Connect),
        success(json!([{"type": "text", "text": "hi"}]), "end_turn"),
    ]);
    let started = Instant::now();
    let response = provider(&transport).generate(user("hi")).await.unwrap();
    assert_eq!(response.content, "hi");
    assert_eq!(transport.requests().len(), 3);
    assert_eq!(started.elapsed(), Duration::from_millis(1_500));
    let bodies = transport.bodies();
    assert!(bodies.windows(2).all(|pair| pair[0] == pair[1]));
}

#[tokio::test(start_paused = true)]
async fn server_retry_hints_override_the_default_policy() {
    let transport = ScriptedTransport::new([
        reply_with(429, &[("retry-after", "3")], json!({})),
        reply_with(
            400,
            &[("x-should-retry", "true"), ("retry-after-ms", "250")],
            json!({}),
        ),
        success(json!([]), "end_turn"),
    ]);
    let started = Instant::now();
    provider(&transport).generate(user("hi")).await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_millis(3_250));

    let transport =
        ScriptedTransport::new([reply_with(503, &[("x-should-retry", "false")], json!({}))]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert_eq!(transport.requests().len(), 1);
    assert!(matches!(error, ProviderError::Api { status: 503, .. }));
}

#[tokio::test(start_paused = true)]
async fn retries_stop_after_the_configured_limit() {
    let transport = ScriptedTransport::new([
        Step::Fail(ErrorKind::Timeout),
        Step::Fail(ErrorKind::Connect),
    ]);
    let error = provider_with(&transport, |config| config.max_retries = 1)
        .generate(user("hi"))
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 2);
    let ProviderError::Transport(source) = error else {
        panic!("expected a transport error");
    };
    let source = source.downcast_ref::<HttpError>().unwrap();
    assert_eq!(source.kind(), ErrorKind::Connect);

    let transport = ScriptedTransport::new([Step::Fail(ErrorKind::Request)]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert!(matches!(error, ProviderError::Transport(_)));
    assert_eq!(transport.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn each_attempt_is_bounded_by_the_configured_timeout() {
    let transport = ScriptedTransport::new([Step::Hang, Step::Hang]);
    let started = Instant::now();
    let error = provider_with(&transport, |config| {
        config.timeout = Duration::from_secs(10);
        config.max_retries = 1;
    })
    .generate(user("hi"))
    .await
    .unwrap_err();
    assert!(matches!(error, ProviderError::Timeout(timeout) if timeout == Duration::from_secs(10)));
    assert_eq!(transport.requests().len(), 2);
    assert_eq!(started.elapsed(), Duration::from_millis(20_500));
}

#[test]
fn configuration_is_validated_and_the_api_key_is_redacted() {
    let transport: Arc<ScriptedTransport> = ScriptedTransport::new([]);
    let build = |configure: fn(&mut AnthropicConfig)| {
        let mut config = AnthropicConfig::new("sk-secret");
        configure(&mut config);
        AnthropicProvider::new(config, transport.clone())
    };
    assert!(matches!(
        build(|config| config.compaction_trigger_tokens = 49_999),
        Err(Error::CompactionTriggerTooLow { minimum: 50_000 })
    ));
    assert!(matches!(
        build(|config| config.base_url = "not a url".into()),
        Err(Error::InvalidBaseUrl)
    ));
    let invalid_key = AnthropicProvider::new(AnthropicConfig::new("bad\nkey"), transport.clone());
    assert!(matches!(invalid_key, Err(Error::InvalidApiKey)));

    let provider = build(|_| {}).unwrap();
    assert_eq!(provider.name(), "anthropic");
    assert_eq!(provider.model_limits(None).context_window_tokens(), 200_000);
    assert_eq!(provider.model_limits(None).max_output_tokens(), 4_096);
    let debug = format!("{provider:?}");
    assert!(!debug.contains("sk-secret"), "{debug}");
    assert!(debug.contains("<redacted>"));
}
