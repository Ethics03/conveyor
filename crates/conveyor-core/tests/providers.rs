use std::sync::Arc;

use conveyor_core::{
    Error,
    models::{JsonObject, MessageRole, ToolCall},
    providers::{
        FinishReason, ModelLimits, Provider, ProviderError, ProviderMessage, ProviderRequest,
        ProviderResponse, ReplayState,
        fake::{DEFAULT_FAKE_MODEL_LIMITS, FakeProvider},
    },
};
use serde_json::{Value, json};

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

#[test]
fn model_limits_reserve_output_inside_a_positive_window() {
    let limits = ModelLimits::new(200_000, 4_096).unwrap();
    assert_eq!(limits.context_window_tokens(), 200_000);
    assert_eq!(limits.max_output_tokens(), 4_096);
    assert!(matches!(
        ModelLimits::new(0, 1),
        Err(Error::EmptyContextWindow)
    ));
    assert!(matches!(
        ModelLimits::new(10, 0),
        Err(Error::EmptyOutputLimit)
    ));
    assert!(matches!(
        ModelLimits::new(10, 10),
        Err(Error::OutputExceedsContextWindow)
    ));
}

#[test]
fn finish_reasons_keep_their_wire_spellings() {
    let reasons = [
        FinishReason::Stop,
        FinishReason::ToolUse,
        FinishReason::MaxTokens,
        FinishReason::Refusal,
        FinishReason::Pause,
        FinishReason::Unknown,
    ];
    let spellings: Vec<_> = reasons.iter().map(|reason| reason.as_str()).collect();
    assert_eq!(
        spellings,
        [
            "stop",
            "tool_use",
            "max_tokens",
            "refusal",
            "pause",
            "unknown"
        ]
    );
    assert_eq!(serde_json::to_value(reasons).unwrap(), json!(spellings));
}

#[test]
fn replay_state_round_trips_the_persisted_metadata_shape() {
    let metadata = json!({
        "provider": "anthropic",
        "items": [{"type": "compaction", "content": "summary", "encrypted_content": "opaque"}],
    });
    let state = ReplayState::from_metadata(&metadata).unwrap();
    assert_eq!(state.provider, "anthropic");
    assert_eq!(state.items[0]["content"], "summary");
    assert_eq!(state.to_metadata(), metadata);
    assert_eq!(ReplayState::METADATA_KEY, "provider_replay_state");
}

#[test]
fn malformed_replay_metadata_is_ignored() {
    for value in [
        json!(null),
        json!("anthropic"),
        json!({"provider": 1, "items": []}),
        json!({"provider": "anthropic"}),
        json!({"provider": "anthropic", "items": {}}),
        json!({"provider": "anthropic", "items": ["not an object"]}),
    ] {
        assert_eq!(ReplayState::from_metadata(&value), None, "{value}");
    }
}

#[test]
fn provider_messages_expose_roles_and_serialize_with_a_role_tag() {
    let call = ToolCall {
        id: "call_1".into(),
        name: "read_file".into(),
        arguments: object(json!({"path": "README.md"})),
    };
    let messages = [
        ProviderMessage::system("Be brief"),
        ProviderMessage::user("hi"),
        ProviderMessage::Assistant {
            content: String::new(),
            tool_calls: vec![call],
            replay_state: None,
        },
        ProviderMessage::tool_result("call_1", "read_file", "denied", true),
    ];
    let roles: Vec<_> = messages.iter().map(ProviderMessage::role).collect();
    assert_eq!(
        roles,
        [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Tool
        ]
    );
    assert_eq!(messages[3].content(), "denied");

    let encoded = serde_json::to_value(&messages).unwrap();
    assert_eq!(
        encoded,
        json!([
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "name": "read_file", "arguments": {"path": "README.md"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
             "content": "denied", "is_error": true},
        ])
    );
    let decoded: Vec<ProviderMessage> = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, messages);
}

#[test]
fn response_constructors_set_matching_finish_reasons() {
    let message = ProviderResponse::from("done");
    assert_eq!(message.content, "done");
    assert_eq!(message.finish_reason, FinishReason::Stop);
    assert!(message.tool_calls.is_empty());

    let call = ToolCall::new("bash", JsonObject::new());
    let tool = ProviderResponse::tool_call(call.clone());
    assert_eq!(tool.content, "");
    assert_eq!(tool.tool_calls, [call]);
    assert_eq!(tool.finish_reason, FinishReason::ToolUse);
}

#[tokio::test]
async fn the_fake_provider_replays_its_script_then_answers_ok() {
    let provider = FakeProvider::with_results([
        Ok(ProviderResponse::message("first")),
        Err(ProviderError::InvalidResponse("scripted".into())),
    ]);
    assert_eq!(provider.model_limits(None), DEFAULT_FAKE_MODEL_LIMITS);
    let request = ProviderRequest::new(vec![ProviderMessage::user("hi")]);

    assert_eq!(
        provider.generate(request.clone()).await.unwrap().content,
        "first"
    );
    let error = provider.generate(request.clone()).await.unwrap_err();
    assert!(matches!(error, ProviderError::InvalidResponse(message) if message == "scripted"));
    assert_eq!(
        provider.generate(request.clone()).await.unwrap().content,
        "ok"
    );
    assert_eq!(
        provider.requests(),
        [request.clone(), request.clone(), request]
    );
}

#[tokio::test]
async fn a_dropped_fake_request_neither_records_nor_consumes_its_step() {
    let provider = FakeProvider::new(["kept"]);
    let request = ProviderRequest::new(vec![ProviderMessage::user("hi")]);
    drop(provider.generate(request.clone()));
    assert!(provider.requests().is_empty());
    assert_eq!(provider.generate(request).await.unwrap().content, "kept");
}

#[tokio::test]
async fn the_fake_provider_is_usable_as_a_shared_trait_object() {
    let limits = ModelLimits::new(1_000, 100).unwrap();
    let provider: Arc<dyn Provider> = Arc::new(FakeProvider::default().with_model_limits(limits));
    assert_eq!(provider.name(), "fake");
    assert_eq!(provider.model_limits(Some("any")), limits);

    let request = ProviderRequest::new(vec![ProviderMessage::user("hi")]);
    let response = provider.generate(request).await.unwrap();
    assert_eq!(response.content, "ok");
    assert_eq!(response.finish_reason, FinishReason::Stop);
}
