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
        openai::{OpenAiConfig, OpenAiProvider, ReasoningEffort},
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

impl std::fmt::Debug for Step {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Step")
    }
}

#[derive(Debug, Default)]
struct ScriptedTransport {
    steps: Mutex<VecDeque<Step>>,
    requests: Mutex<Vec<Request<Bytes>>>,
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

/// A response body with `output` items and the given status.
fn response_body(status: &str, output: Value) -> Value {
    json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-5.5-2026-04-23",
        "status": status,
        "incomplete_details": null,
        "error": null,
        "output": output,
        "usage": {
            "input_tokens": 40,
            "input_tokens_details": {"cached_tokens": 32},
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 8},
            "total_tokens": 52,
        },
    })
}

fn success(output: Value) -> Step {
    reply(200, response_body("completed", output))
}

fn message_item(text: &str) -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    })
}

fn reasoning_item(id: &str, encrypted: Option<&str>) -> Value {
    let mut item = json!({"id": id, "type": "reasoning", "summary": [], "content": []});
    if let Some(encrypted) = encrypted {
        item["encrypted_content"] = encrypted.into();
    }
    item
}

fn provider(transport: &Arc<ScriptedTransport>) -> OpenAiProvider {
    provider_with(transport, |_| {})
}

fn provider_with(
    transport: &Arc<ScriptedTransport>,
    configure: impl FnOnce(&mut OpenAiConfig),
) -> OpenAiProvider {
    let mut config = OpenAiConfig::new("sk-test");
    configure(&mut config);
    OpenAiProvider::new(config, transport.clone()).unwrap()
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

fn replay(provider: &str, items: Vec<Value>) -> ReplayState {
    ReplayState {
        provider: provider.into(),
        items: items.into_iter().map(object).collect(),
    }
}

#[tokio::test]
async fn requests_encode_instructions_messages_tools_and_headers() {
    let transport = ScriptedTransport::new([success(json!([message_item("Done.")]))]);
    let provider = provider_with(&transport, |config| {
        config.organization = Some("org_1".into());
        config.project = Some("proj_1".into());
    });
    let call = ToolCall {
        id: "call_1".into(),
        name: "read_file".into(),
        arguments: object(json!({"path": "Cargo.toml"})),
    };
    let mut request = ProviderRequest::new(vec![
        ProviderMessage::system("You are careful."),
        ProviderMessage::system("Temporal context (UTC)"),
        ProviderMessage::user("Read the manifest."),
        ProviderMessage::Assistant {
            content: "Reading it.".into(),
            tool_calls: vec![call],
            replay_state: None,
        },
        ProviderMessage::tool_result("call_1", "read_file", "[workspace]", false),
        ProviderMessage::system("Be brief."),
        ProviderMessage::tool_result("call_2", "read_file", "Not a file: x", true),
    ]);
    request.tools = vec![
        ToolDefinition::new(
            "read_file",
            "Read a file",
            object(json!({"type": "object", "properties": {"path": {"type": "string"}}})),
        ),
        ToolDefinition::new("noop", "", JsonObject::new()),
    ];

    let response = provider.generate(request).await.unwrap();

    assert_eq!(response.content, "Done.");
    let sent = &transport.requests()[0];
    assert_eq!(sent.uri(), "https://api.openai.com/v1/responses");
    assert_eq!(sent.headers()["authorization"], "Bearer sk-test");
    assert!(sent.headers()["authorization"].is_sensitive());
    assert_eq!(sent.headers()["content-type"], "application/json");
    assert_eq!(sent.headers()["openai-organization"], "org_1");
    assert_eq!(sent.headers()["openai-project"], "proj_1");
    assert_eq!(
        transport.bodies()[0],
        json!({
            "model": "gpt-5.5",
            "instructions": "You are careful.\n\nTemporal context (UTC)",
            "input": [
                {"role": "user", "content": "Read the manifest."},
                {"role": "assistant", "content": "Reading it."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{\"path\":\"Cargo.toml\"}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "[workspace]"},
                {"role": "developer", "content": "Be brief."},
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "Error: Not a file: x",
                },
            ],
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 128_000,
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "strict": false,
                },
                {
                    "type": "function",
                    "name": "noop",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": false,
                },
            ],
        })
    );
}

#[tokio::test]
async fn request_options_override_the_defaults() {
    let transport = ScriptedTransport::new([success(json!([]))]);
    let provider = provider_with(&transport, |config| {
        config.reasoning_effort = Some(ReasoningEffort::XHigh);
        config.base_url = "https://gateway.example/openai/v1/".into();
    });
    let mut request = user("hi");
    request.model = Some("gpt-5.3-codex".into());
    request.max_tokens = Some(2_000);
    request.temperature = Some(0.2);

    provider.generate(request).await.unwrap();

    let body = &transport.bodies()[0];
    assert_eq!(
        transport.requests()[0].uri(),
        "https://gateway.example/openai/v1/responses"
    );
    assert_eq!(body["model"], "gpt-5.3-codex");
    assert_eq!(body["max_output_tokens"], 2_000);
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["reasoning"], json!({"effort": "xhigh"}));
    assert!(body.get("instructions").is_none());
    assert!(body.get("tools").is_none());

    let mut invalid = user("hi");
    invalid.temperature = Some(f64::NAN);
    let error = provider.generate(invalid).await.unwrap_err();
    assert!(matches!(error, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn system_only_and_empty_conversations_still_send_input() {
    let transport = ScriptedTransport::new([success(json!([])), success(json!([]))]);
    let provider = provider(&transport);

    let only_system = ProviderRequest::new(vec![ProviderMessage::system("Summarize.")]);
    provider.generate(only_system).await.unwrap();
    provider
        .generate(ProviderRequest::new(Vec::new()))
        .await
        .unwrap();

    let bodies = transport.bodies();
    assert!(bodies[0].get("instructions").is_none());
    assert_eq!(
        bodies[0]["input"],
        json!([{"role": "developer", "content": "Summarize."}])
    );
    assert_eq!(bodies[1]["input"], json!([{"role": "user", "content": ""}]));
}

#[tokio::test]
async fn saved_output_items_are_replayed_in_order() {
    let transport = ScriptedTransport::new([success(json!([]))]);
    let items = vec![
        reasoning_item("rs_1", Some("ENC_1")),
        json!({
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "search_files",
            "arguments": "{\"pattern\":\"main\"}",
        }),
    ];
    let request = ProviderRequest::new(vec![
        ProviderMessage::user("Find main."),
        ProviderMessage::Assistant {
            // Ignored in favor of the saved items.
            content: "rebuilt text".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "search_files".into(),
                arguments: object(json!({"pattern": "main"})),
            }],
            replay_state: Some(replay("openai", items.clone())),
        },
        ProviderMessage::tool_result("call_1", "search_files", "src/main.rs", false),
    ]);

    provider(&transport).generate(request).await.unwrap();

    let input = transport.bodies()[0]["input"].clone();
    assert_eq!(
        input,
        json!([
            {"role": "user", "content": "Find main."},
            items[0],
            items[1],
            {"type": "function_call_output", "call_id": "call_1", "output": "src/main.rs"},
        ])
    );
}

#[tokio::test]
async fn unresolvable_reasoning_is_dropped_with_its_pairing() {
    let transport = ScriptedTransport::new([success(json!([]))]);
    let request = ProviderRequest::new(vec![
        ProviderMessage::user("Plan it."),
        ProviderMessage::Assistant {
            content: String::new(),
            tool_calls: Vec::new(),
            replay_state: Some(replay(
                "openai",
                vec![
                    reasoning_item("rs_1", None),
                    message_item("Here is the plan."),
                    reasoning_item("rs_2", Some("")),
                    json!({
                        "id": "fc_2",
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "read_file",
                        "arguments": "{}",
                    }),
                    reasoning_item("rs_3", Some("ENC_3")),
                    message_item("Kept."),
                ],
            )),
        },
    ]);

    provider(&transport).generate(request).await.unwrap();

    assert_eq!(
        transport.bodies()[0]["input"],
        json!([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": "Here is the plan."},
            {"type": "function_call", "call_id": "call_2", "name": "read_file", "arguments": "{}"},
            reasoning_item("rs_3", Some("ENC_3")),
            message_item("Kept."),
        ])
    );
}

#[tokio::test]
async fn replay_state_from_another_provider_is_rebuilt_as_text() {
    let transport = ScriptedTransport::new([success(json!([]))]);
    let request = ProviderRequest::new(vec![
        ProviderMessage::user("hi"),
        ProviderMessage::Assistant {
            content: "hello".into(),
            tool_calls: Vec::new(),
            replay_state: Some(replay(
                "anthropic",
                vec![json!({"type": "compaction", "content": "summary"})],
            )),
        },
    ]);

    provider(&transport).generate(request).await.unwrap();

    assert_eq!(
        transport.bodies()[0]["input"][1],
        json!({"role": "assistant", "content": "hello"})
    );
}

#[tokio::test]
async fn responses_decode_text_tool_calls_usage_and_replay_state() {
    let output = json!([
        reasoning_item("rs_1", Some("ENC_1")),
        message_item("Checking the manifest."),
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_abc",
            "name": "read_file",
            "arguments": "{\"path\":\"Cargo.toml\",\"limit\":20}",
        },
        {
            "id": "fc_2",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_def",
            "name": "list",
            "arguments": "",
        },
    ]);
    let transport = ScriptedTransport::new([success(output.clone())]);

    let response = provider(&transport).generate(user("hi")).await.unwrap();

    assert_eq!(response.content, "Checking the manifest.");
    assert_eq!(response.finish_reason, FinishReason::ToolUse);
    assert_eq!(
        response.tool_calls,
        [
            ToolCall {
                id: "call_abc".into(),
                name: "read_file".into(),
                arguments: object(json!({"path": "Cargo.toml", "limit": 20})),
            },
            ToolCall {
                id: "call_def".into(),
                name: "list".into(),
                arguments: JsonObject::new(),
            },
        ]
    );
    assert_eq!(
        Value::Object(response.raw),
        json!({
            "provider": "openai",
            "model": "gpt-5.5-2026-04-23",
            "response_id": "resp_1",
            "usage": {
                "input_tokens": 40,
                "input_tokens_details": {"cached_tokens": 32},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 52,
            },
        })
    );
    let state = response.replay_state.unwrap();
    assert_eq!(state.provider, "openai");
    assert_eq!(
        Value::Array(state.items.into_iter().map(Value::Object).collect()),
        output
    );
}

#[tokio::test]
async fn statuses_and_incomplete_reasons_map_to_finish_reasons() {
    let refusal = json!([{
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }]);
    let incomplete = |reason: Value| {
        let mut body = response_body("incomplete", json!([message_item("Partial")]));
        body["incomplete_details"] = json!({"reason": reason});
        reply(200, body)
    };
    let cases = [
        (success(json!([message_item("ok")])), FinishReason::Stop),
        (success(refusal), FinishReason::Refusal),
        (
            incomplete(json!("max_output_tokens")),
            FinishReason::MaxTokens,
        ),
        (incomplete(json!("content_filter")), FinishReason::Refusal),
        (incomplete(json!("something_new")), FinishReason::Unknown),
        (incomplete(Value::Null), FinishReason::Unknown),
    ];
    for (step, expected) in cases {
        let transport = ScriptedTransport::new([step]);
        let response = provider(&transport).generate(user("hi")).await.unwrap();
        assert_eq!(response.finish_reason, expected);
        assert!(!response.content.is_empty());
    }

    // A response with no status, as some compatible gateways send.
    let mut body = response_body("completed", json!([message_item("ok")]));
    body.as_object_mut().unwrap().remove("status");
    let transport = ScriptedTransport::new([reply(200, body)]);
    let response = provider(&transport).generate(user("hi")).await.unwrap();
    assert_eq!(response.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn failed_and_unfinished_responses_are_errors() {
    let mut failed = response_body("failed", json!([]));
    failed["error"] = json!({"code": "server_error", "message": "The model crashed."});
    let cases = [
        (failed, "Response failed: server_error: The model crashed."),
        (response_body("cancelled", json!([])), "Response cancelled"),
        (
            response_body("in_progress", json!([])),
            "Response ended with status in_progress",
        ),
    ];
    for (body, message) in cases {
        let transport = ScriptedTransport::new([reply(200, body)]);
        let error = provider(&transport).generate(user("hi")).await.unwrap_err();
        assert!(
            matches!(&error, ProviderError::InvalidResponse(text) if text == message),
            "{error:?}"
        );
    }
}

#[tokio::test]
async fn malformed_function_calls_fail_unless_the_response_was_cut_short() {
    let call = |arguments: &str| {
        json!([{"type": "function_call", "call_id": "call_1", "name": "read_file",
                "arguments": arguments}])
    };
    for arguments in ["{\"path\":", "[1, 2]"] {
        let transport = ScriptedTransport::new([success(call(arguments))]);
        let error = provider(&transport).generate(user("hi")).await.unwrap_err();
        assert!(
            matches!(error, ProviderError::InvalidResponse(_)),
            "{arguments}"
        );
    }
    let transport = ScriptedTransport::new([success(
        json!([{"type": "function_call", "name": "read_file", "arguments": "{}"}]),
    )]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert!(
        matches!(&error, ProviderError::InvalidResponse(text) if text == "Function call read_file has no call_id")
    );

    let mut body = response_body("incomplete", call("{\"path\":"));
    body["incomplete_details"] = json!({"reason": "max_output_tokens"});
    let transport = ScriptedTransport::new([reply(200, body)]);
    let response = provider(&transport).generate(user("hi")).await.unwrap();
    assert!(response.tool_calls.is_empty());
    assert_eq!(response.finish_reason, FinishReason::MaxTokens);
}

#[tokio::test]
async fn malformed_success_bodies_are_invalid_responses() {
    for body in [json!({"unexpected": true}), json!("not an object")] {
        let transport = ScriptedTransport::new([reply(200, body)]);
        let error = provider(&transport).generate(user("hi")).await.unwrap_err();
        assert!(matches!(error, ProviderError::InvalidResponse(_)));
    }
}

#[tokio::test]
async fn client_errors_are_reported_without_retrying() {
    let transport = ScriptedTransport::new([reply_with(
        400,
        &[("x-request-id", "req_1")],
        json!({"error": {
            "message": "Your input exceeds the context window of this model.",
            "type": "invalid_request_error",
            "param": "input",
            "code": "context_length_exceeded",
        }}),
    )]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert_eq!(transport.requests().len(), 1);
    let ProviderError::Api {
        status,
        kind,
        message,
        request_id,
    } = error
    else {
        panic!("expected an API error");
    };
    assert_eq!(status, 400);
    assert_eq!(kind.as_deref(), Some("context_length_exceeded"));
    assert_eq!(
        message,
        "Your input exceeds the context window of this model."
    );
    assert_eq!(request_id.as_deref(), Some("req_1"));

    // Without a code, the type names the error; without a body, the status.
    let transport = ScriptedTransport::new([
        reply(
            403,
            json!({"error": {"message": "No access", "type": "invalid_request_error"}}),
        ),
        Step::Reply(Response::builder().status(404).body(Bytes::new()).unwrap()),
    ]);
    let provider = provider(&transport);
    let error = provider.generate(user("hi")).await.unwrap_err();
    assert!(
        matches!(&error, ProviderError::Api { kind: Some(kind), .. } if kind == "invalid_request_error")
    );
    let error = provider.generate(user("hi")).await.unwrap_err();
    assert!(
        matches!(&error, ProviderError::Api { status: 404, kind: None, message, .. } if message == "Not Found")
    );
}

#[tokio::test(start_paused = true)]
async fn billing_errors_are_final_even_when_rate_limited() {
    for body in [
        json!({"error": {"message": "You exceeded your current quota.",
                         "type": "insufficient_quota", "code": "insufficient_quota"}}),
        json!({"error": {"message": "Credit balance exhausted.",
                         "type": "insufficient_quota", "code": "credit_balance_exhausted"}}),
    ] {
        let transport = ScriptedTransport::new([reply_with(429, &[("retry-after", "1")], body)]);
        let error = provider(&transport).generate(user("hi")).await.unwrap_err();
        assert_eq!(transport.requests().len(), 1);
        assert!(matches!(error, ProviderError::Api { status: 429, .. }));
    }
}

#[tokio::test(start_paused = true)]
async fn rate_limits_and_server_errors_retry_with_hints_or_backoff() {
    let transport = ScriptedTransport::new([
        reply(
            429,
            json!({"error": {"message": "Rate limit reached for gpt-5.5. Please try again in 1.5s.",
                             "type": "requests", "code": "rate_limit_exceeded"}}),
        ),
        reply_with(429, &[("retry-after-ms", "250")], json!({})),
        reply(
            500,
            json!({"error": {"message": "server error", "type": "server_error"}}),
        ),
        Step::Fail(ErrorKind::Interrupted),
        success(json!([message_item("hi")])),
    ]);
    let started = Instant::now();

    let response = provider_with(&transport, |config| config.max_retries = 4)
        .generate(user("hi"))
        .await
        .unwrap();

    assert_eq!(response.content, "hi");
    assert_eq!(transport.requests().len(), 5);
    // 1.5 s hinted, 0.25 s from the header, then backoff of 2 s and 4 s.
    assert_eq!(started.elapsed(), Duration::from_millis(7_750));
    let bodies = transport.bodies();
    assert!(bodies.windows(2).all(|pair| pair[0] == pair[1]));
}

#[tokio::test(start_paused = true)]
async fn retries_stop_at_the_limit_and_attempts_time_out() {
    let transport = ScriptedTransport::new([
        Step::Fail(ErrorKind::Connect),
        Step::Fail(ErrorKind::Connect),
        Step::Fail(ErrorKind::Connect),
    ]);
    let error = provider(&transport).generate(user("hi")).await.unwrap_err();
    assert_eq!(transport.requests().len(), 3);
    assert!(matches!(error, ProviderError::Transport(_)));

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
    assert_eq!(started.elapsed(), Duration::from_millis(20_500));
}

#[test]
fn configuration_is_validated_and_secrets_are_redacted() {
    let transport: Arc<ScriptedTransport> = ScriptedTransport::new([]);
    let build = |configure: fn(&mut OpenAiConfig)| {
        let mut config = OpenAiConfig::new("sk-secret");
        configure(&mut config);
        OpenAiProvider::new(config, transport.clone())
    };
    assert!(matches!(
        build(|config| config.base_url = "not a url".into()),
        Err(Error::InvalidBaseUrl)
    ));
    assert!(matches!(
        build(|config| config.organization = Some("org\n1".into())),
        Err(Error::InvalidHeaderValue("OpenAI organization"))
    ));
    assert!(matches!(
        build(|config| config.project = Some("proj\n1".into())),
        Err(Error::InvalidHeaderValue("OpenAI project"))
    ));
    assert!(matches!(
        OpenAiProvider::new(OpenAiConfig::new("bad\nkey"), transport.clone()),
        Err(Error::InvalidApiKey)
    ));

    let provider = build(|_| {}).unwrap();
    assert_eq!(provider.name(), "openai");
    assert_eq!(provider.config().model, "gpt-5.5");
    assert_eq!(
        provider.model_limits(None).context_window_tokens(),
        1_050_000
    );
    assert_eq!(provider.model_limits(None).max_output_tokens(), 128_000);
    let debug = format!("{provider:?}");
    assert!(!debug.contains("sk-secret"), "{debug}");
    assert!(debug.contains("<redacted>"));
}
