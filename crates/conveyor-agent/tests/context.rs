use chrono::{DateTime, FixedOffset};
use conveyor_agent::{Error, build_provider_messages};
use conveyor_core::{
    models::{Agent, JsonObject, Message, MessageRole, Run, Session, ToolCall},
    providers::{ProviderMessage, ReplayState},
};
use serde_json::{Value, json};

fn at(timestamp: &str) -> DateTime<FixedOffset> {
    DateTime::parse_from_rfc3339(timestamp).unwrap()
}

fn message(role: MessageRole, content: &str) -> Message {
    Message::new("ses_test", role, content)
}

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

#[test]
fn instructions_come_first_and_empty_instructions_are_omitted() {
    let messages = [
        message(MessageRole::User, "Hello"),
        message(MessageRole::Assistant, "Hi"),
    ];
    let agent = Agent {
        instructions: "Be precise.".into(),
        ..Agent::default()
    };
    assert_eq!(
        build_provider_messages(&agent, &messages, None).unwrap(),
        [
            ProviderMessage::system("Be precise."),
            ProviderMessage::user("Hello"),
            ProviderMessage::assistant("Hi"),
        ]
    );
    assert_eq!(
        build_provider_messages(&Agent::default(), &messages[..1], None).unwrap(),
        [ProviderMessage::user("Hello")]
    );
}

#[test]
fn temporal_context_uses_python_isoformat_and_is_stable_across_runs() {
    let session = Session {
        created_at: at("2026-08-10T08:00:00+00:00"),
        ..Session::default()
    };
    let mut user = message(MessageRole::User, "What happened?");
    user.created_at = at("2026-08-10T08:04:00.123000+00:00");
    let first_run = Run::new(&session.id, "agent");
    let mut later_run = Run::new(&session.id, "agent");
    later_run.created_at = at("2026-08-10T09:00:00+00:00");

    let first = build_provider_messages(
        &Agent::default(),
        std::slice::from_ref(&user),
        Some((&session, &first_run)),
    )
    .unwrap();
    assert_eq!(
        first,
        [
            ProviderMessage::system(
                "Temporal context (UTC):\n- Session created at: 2026-08-10T08:00:00+00:00"
            ),
            ProviderMessage::user(
                "Message timestamp (UTC): 2026-08-10T08:04:00.123000+00:00\n\nWhat happened?"
            ),
        ]
    );
    let later = build_provider_messages(
        &Agent::default(),
        std::slice::from_ref(&user),
        Some((&session, &later_run)),
    )
    .unwrap();
    assert_eq!(first, later);
}

#[test]
fn a_run_from_another_session_is_rejected() {
    let session = Session::default();
    let run = Run::new(&Session::default().id, "agent");
    let error =
        build_provider_messages(&Agent::default(), &[], Some((&session, &run))).unwrap_err();
    assert!(matches!(error, Error::RunSessionMismatch));
}

#[test]
fn tool_exchanges_keep_their_correlation_failure_flags_and_replay_state() {
    let call = ToolCall {
        id: "call_readme".into(),
        name: "read_file".into(),
        arguments: object(json!({"path": "README.md"})),
    };
    let state = json!({"provider": "anthropic", "items": [{"type": "compaction", "content": "s"}]});
    let mut assistant = message(MessageRole::Assistant, "");
    assistant.tool_calls = vec![call.clone()];
    assistant
        .metadata
        .insert(ReplayState::METADATA_KEY.into(), state.clone());
    let mut failed = message(MessageRole::Tool, "Permission denied");
    failed.tool_call_id = Some(call.id.clone());
    failed.name = Some("read_file".into());
    failed.metadata.insert("ok".into(), false.into());
    let mut unnamed = message(MessageRole::Tool, "fine");
    unnamed.tool_call_id = Some("call_other".into());
    unnamed.metadata.insert("ok".into(), true.into());

    let converted =
        build_provider_messages(&Agent::default(), &[assistant, failed, unnamed], None).unwrap();
    assert_eq!(
        converted,
        [
            ProviderMessage::Assistant {
                content: String::new(),
                tool_calls: vec![call],
                replay_state: ReplayState::from_metadata(&state),
            },
            ProviderMessage::tool_result("call_readme", "read_file", "Permission denied", true),
            ProviderMessage::tool_result("call_other", "tool", "fine", false),
        ]
    );
}

#[test]
fn tool_messages_without_a_call_id_are_rejected() {
    let orphan = message(MessageRole::Tool, "result");
    let error = build_provider_messages(&Agent::default(), std::slice::from_ref(&orphan), None)
        .unwrap_err();
    assert!(matches!(error, Error::MissingToolCallId(id) if id == orphan.id));
}
