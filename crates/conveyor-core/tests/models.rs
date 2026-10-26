use conveyor_core::{
    Error,
    models::{
        Agent, ApprovalDecision, ApprovalRequest, ApprovalStatus, Event, EventType, JsonObject,
        Message, MessageRole, Run, RunStatus, Session, SessionStatus, SessionTitleSource, ToolCall,
        ToolPermission, clean_title,
    },
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use uuid::{Uuid, Version};

fn wire_values<T: DeserializeOwned + Serialize>(fixture: &Value, name: &str) {
    let values: Vec<T> = serde_json::from_value(fixture[name].clone()).unwrap();
    assert_eq!(serde_json::to_value(values).unwrap(), fixture[name]);
}

#[test]
fn wire_enums_match_the_storage_format_contracts() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../test-support/fixtures/storage-v1/records.json"
    ))
    .unwrap();
    let values = &fixture["wire_values"];
    wire_values::<SessionStatus>(values, "SessionStatus");
    wire_values::<SessionTitleSource>(values, "SessionTitleSource");
    wire_values::<RunStatus>(values, "RunStatus");
    wire_values::<MessageRole>(values, "MessageRole");
    wire_values::<EventType>(values, "EventType");
    wire_values::<ApprovalStatus>(values, "ApprovalStatus");
    wire_values::<ApprovalDecision>(values, "ApprovalDecision");
    wire_values::<ToolPermission>(values, "ToolPermission");
    assert!(serde_json::from_str::<ApprovalDecision>("\"pending\"").is_err());
    assert!(serde_json::from_str::<RunStatus>("\"invented_status\"").is_err());
}

#[test]
fn generated_ids_are_prefixed_uuid7_and_constructor_dates_have_microsecond_precision() {
    let session = Session::default();
    let agent = Agent::default();
    let run = Run::new(&session.id, &agent.id);
    let message = Message::new(&session.id, MessageRole::User, "hello");
    let event = Event::new(EventType::MessageCreated);
    let call = ToolCall::new("read_file", JsonObject::new());
    let approval = ApprovalRequest::new(&session.id, &run.id, call.clone());
    for (prefix, id) in [
        ("ses", &session.id),
        ("agent", &agent.id),
        ("run", &run.id),
        ("msg", &message.id),
        ("evt", &event.id),
        ("call", &call.id),
        ("appr", &approval.id),
    ] {
        let raw = id.strip_prefix(&format!("{prefix}_")).unwrap();
        assert_eq!(raw.len(), 32);
        assert_eq!(
            Uuid::parse_str(raw).unwrap().get_version(),
            Some(Version::SortRand)
        );
    }
    assert_ne!(Session::default().id, session.id);
    assert_eq!(session.created_at.timestamp_subsec_nanos() % 1000, 0);
    assert_eq!(approval.status, ApprovalStatus::Pending);
    assert_eq!(approval.resolved_at, None);
}

#[test]
fn title_validation_counts_unicode_characters_and_normalizes_whitespace() {
    let session = Session::with_title(" \n Fix\u{a0}provider\u{1c}timeouts \t").unwrap();
    assert_eq!(session.title, "Fix provider timeouts");
    assert_eq!(session.title_source, SessionTitleSource::User);
    assert!(Session::with_title(&"🦀".repeat(100)).is_ok());
    assert!(matches!(
        Session::with_title(&"🦀".repeat(101)),
        Err(Error::TitleTooLong)
    ));
    assert!(matches!(
        Session::with_title(" \n \t"),
        Err(Error::EmptyTitle)
    ));
}

#[test]
fn title_limits_apply_after_whitespace_normalization() {
    let prefix = "🦀".repeat(98);
    let title = format!("\u{1c}\u{1d} {prefix}\u{1e}\u{1f}\t x \n");
    assert_eq!(clean_title(&title).unwrap(), format!("{prefix} x"));
    assert!(matches!(
        clean_title(&format!("{title}y")),
        Err(Error::TitleTooLong)
    ));
    assert!(matches!(
        clean_title("\u{1c}\u{1d}\u{1e}\u{1f}"),
        Err(Error::EmptyTitle)
    ));
    assert!(matches!(
        clean_title(&"long title ".repeat(100_000)),
        Err(Error::TitleTooLong)
    ));
}
