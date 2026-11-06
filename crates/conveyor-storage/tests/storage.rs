use conveyor_core::{
    Error as DomainError,
    models::{
        ApprovalDecision, ApprovalRequest, ApprovalStatus, Event, EventType, JsonObject, Message,
        MessageRole, Run, RunStatus, Session, SessionStatus, SessionTitleSource, ToolCall, now,
    },
};
use conveyor_storage::{Error, Store};
use serde_json::json;
use sqlx::{Connection, SqliteConnection, sqlite::SqliteConnectOptions};

#[tokio::test]
async fn unsupported_timestamp_precision_is_rejected_without_a_partial_write() {
    let store = Store::open(":memory:").await.unwrap();
    let session = Session {
        created_at: "2026-10-26T12:00:00.123456789+00:00".parse().unwrap(),
        ..Session::default()
    };
    assert!(matches!(
        store.save_session(&session).await,
        Err(Error::TimestampPrecision)
    ));
    assert!(store.list_sessions(None).await.unwrap().is_empty());
    store.close().await.unwrap();
}

async fn seeded() -> (Store, Session, Run) {
    let store = Store::open(":memory:").await.unwrap();
    let session = Session::default();
    store.save_session(&session).await.unwrap();
    let mut run = Run::new(&session.id, "agent_test");
    run.status = RunStatus::Running;
    store.save_run(&run).await.unwrap();
    (store, session, run)
}

fn request(session: &Session, run: &Run) -> ApprovalRequest {
    ApprovalRequest::new(
        &session.id,
        &run.id,
        ToolCall::new("write_file", JsonObject::new()),
    )
}

#[tokio::test]
async fn sessions_preserve_titles_on_upsert_and_filter_archived_records() {
    let (store, mut session, _) = seeded().await;
    let original = session.clone();
    session.title = "This should not overwrite the saved title".into();
    session.title_source = SessionTitleSource::User;
    session.status = SessionStatus::Archived;
    store.save_session(&session).await.unwrap();
    let loaded = store.get_session(&session.id).await.unwrap().unwrap();
    assert_eq!(loaded.title, original.title);
    assert_eq!(loaded.title_source, original.title_source);
    assert_eq!(loaded.status, SessionStatus::Archived);
    assert_eq!(
        store.list_sessions(None).await.unwrap(),
        std::slice::from_ref(&loaded)
    );
    assert_eq!(
        store
            .list_sessions(Some(SessionStatus::Archived))
            .await
            .unwrap(),
        [loaded]
    );
    assert!(
        store
            .list_sessions(Some(SessionStatus::Active))
            .await
            .unwrap()
            .is_empty()
    );
    assert!(store.get_session("missing").await.unwrap().is_none());
    store.close().await.unwrap();
}

#[tokio::test]
async fn automatic_titles_yield_to_user_titles_and_invalid_updates_do_not_modify_state() {
    let (store, session, _) = seeded().await;
    assert!(
        store
            .set_auto_title(&session.id, "  Fix\n timeout ")
            .await
            .unwrap()
    );
    assert!(
        !store
            .set_auto_title(&session.id, "Another automatic title")
            .await
            .unwrap()
    );
    assert!(
        store
            .set_session_title(&session.id, "My title")
            .await
            .unwrap()
    );
    assert!(
        !store
            .set_auto_title(&session.id, "Late result")
            .await
            .unwrap()
    );
    assert!(!store.set_session_title("missing", "Unknown").await.unwrap());
    assert!(matches!(
        store.set_session_title(&session.id, " ").await,
        Err(Error::Domain(DomainError::EmptyTitle))
    ));
    assert!(matches!(
        store
            .set_session_title(&session.id, &"🦀".repeat(101))
            .await,
        Err(Error::Domain(DomainError::TitleTooLong))
    ));
    let loaded = store.get_session(&session.id).await.unwrap().unwrap();
    assert_eq!(loaded.title, "My title");
    assert_eq!(loaded.title_source, SessionTitleSource::User);
    store.close().await.unwrap();
}

#[tokio::test]
async fn run_upserts_preserve_identity_and_support_parent_relationships() {
    let (store, session, parent) = seeded().await;
    let mut child = Run::new(&session.id, "agent_child");
    child.parent_run_id = Some(parent.id.clone());
    store.save_run(&child).await.unwrap();
    let original = child.clone();
    child.agent_id = "agent_changed".into();
    child.session_id = "missing".into();
    child.parent_run_id = None;
    child.status = RunStatus::Failed;
    child.error = Some("provider error".into());
    // ON CONFLICT only updates lifecycle fields; identity remains unchanged.
    store.save_run(&child).await.unwrap();
    let loaded = store.get_run(&child.id).await.unwrap().unwrap();
    assert_eq!(loaded.agent_id, original.agent_id);
    assert_eq!(loaded.parent_run_id, original.parent_run_id);
    assert_eq!(loaded.session_id, original.session_id);
    assert_eq!(loaded.status, RunStatus::Failed);
    assert_eq!(loaded.error, child.error);
    assert_eq!(
        store.list_runs(&session.id).await.unwrap(),
        [parent, loaded]
    );
    assert!(store.get_run("missing").await.unwrap().is_none());
    store.close().await.unwrap();
}

#[tokio::test]
async fn messages_preserve_tool_correlation_opaque_metadata_and_timestamp_id_order() {
    let (store, session, run) = seeded().await;
    let mut assistant = Message::new(&session.id, MessageRole::Assistant, "Inspecting café 🦀");
    assistant.id = "msg_a".into();
    assistant.run_id = Some(run.id.clone());
    let mut call = ToolCall::new(
        "read_file",
        serde_json::from_value(json!({"path":"你好.md"})).unwrap(),
    );
    call.id = "provider_call_id".into();
    assistant.tool_calls.push(call.clone());
    assistant.metadata = serde_json::from_value(
        json!({"opaque":{"encrypted_content":"secret-fixture","items":[null,true,1.5]}}),
    )
    .unwrap();
    let mut tool = Message::new(&session.id, MessageRole::Tool, "café");
    tool.id = "msg_b".into();
    tool.name = Some(call.name.clone());
    tool.tool_call_id = Some(call.id);
    tool.created_at = assistant.created_at;
    store.save_message(&tool).await.unwrap();
    store.save_message(&assistant).await.unwrap();
    assert_eq!(
        store.list_messages(&session.id).await.unwrap(),
        [assistant, tool]
    );
    assert!(store.list_messages("missing").await.unwrap().is_empty());
    store.close().await.unwrap();
}

#[tokio::test]
async fn state_and_events_roll_back_together_on_duplicate_event_ids() {
    let (store, session, mut run) = seeded().await;
    let mut event = Event::new(EventType::RunStarted);
    event.session_id = Some(session.id.clone());
    event.run_id = Some(run.id.clone());
    store.append_event(&event).await.unwrap();
    run.status = RunStatus::Finished;
    assert!(matches!(
        store.save_run_with_events(&run, &[event.clone()]).await,
        Err(Error::Database(_))
    ));
    assert_eq!(
        store.get_run(&run.id).await.unwrap().unwrap().status,
        RunStatus::Running
    );
    let message = Message::new(&session.id, MessageRole::Assistant, "Must roll back");
    assert!(
        store
            .save_message_with_events(&message, &[event.clone()])
            .await
            .is_err()
    );
    assert!(store.list_messages(&session.id).await.unwrap().is_empty());
    assert_eq!(
        store
            .list_events(Some(&session.id), Some(&run.id))
            .await
            .unwrap(),
        [event]
    );
    assert!(
        store
            .list_events(Some("missing"), Some(&run.id))
            .await
            .unwrap()
            .is_empty()
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn foreign_keys_are_enforced_and_a_failed_operation_does_not_poison_the_worker() {
    let store = Store::open(":memory:").await.unwrap();
    let mut invalid = Event::new(EventType::RunStarted);
    invalid.run_id = Some("missing".into());
    assert!(matches!(
        store.append_event(&invalid).await,
        Err(Error::Database(_))
    ));
    let session = Session::default();
    store.save_session(&session).await.unwrap();
    assert_eq!(store.get_session(&session.id).await.unwrap(), Some(session));
    store.close().await.unwrap();
}

#[tokio::test]
async fn block_run_is_atomic_and_rejects_resolved_new_approvals() {
    let (store, session, mut run) = seeded().await;
    let approval = request(&session, &run);
    run.status = RunStatus::Blocked;
    let mut event = Event::new(EventType::RunBlocked);
    event.run_id = Some(run.id.clone());
    assert!(
        store
            .block_run(
                &run,
                std::slice::from_ref(&approval),
                &[event.clone(), event]
            )
            .await
            .is_err()
    );
    assert_eq!(
        store.get_run(&run.id).await.unwrap().unwrap().status,
        RunStatus::Running
    );
    assert!(store.list_approvals(None, None).await.unwrap().is_empty());
    assert!(store.list_events(None, None).await.unwrap().is_empty());
    for status in [ApprovalStatus::Approved, ApprovalStatus::Denied] {
        let mut invalid = approval.clone();
        invalid.status = status;
        assert!(matches!(
            store.save_approval(&invalid).await,
            Err(Error::InvalidNewApproval)
        ));
    }
    let mut invalid = approval.clone();
    invalid.resolved_at = Some(approval.created_at);
    assert!(matches!(
        store.save_approval(&invalid).await,
        Err(Error::InvalidNewApproval)
    ));
    store
        .block_run(&run, std::slice::from_ref(&approval), &[])
        .await
        .unwrap();
    assert_eq!(
        store.get_approval(&approval.id).await.unwrap(),
        Some(approval)
    );
    assert_eq!(store.get_run(&run.id).await.unwrap(), Some(run));
    store.close().await.unwrap();
}

#[tokio::test]
async fn resuming_a_run_uses_compare_and_set_and_rolls_back_on_event_failure() {
    let (store, session, mut run) = seeded().await;
    let mut event = Event::new(EventType::RunResumed);
    event.run_id = Some(run.id.clone());
    assert!(matches!(
        store.resume_run(&run, &event).await,
        Err(Error::RunNotBlocked(_))
    ));
    run.status = RunStatus::Blocked;
    store.save_run(&run).await.unwrap();
    assert!(matches!(
        store.resume_run(&run, &event).await,
        Err(Error::InvalidResumeStatus)
    ));
    run.status = RunStatus::Running;
    event.session_id = Some("missing".into());
    assert!(store.resume_run(&run, &event).await.is_err());
    assert_eq!(
        store.get_run(&run.id).await.unwrap().unwrap().status,
        RunStatus::Blocked
    );
    event.session_id = Some(session.id);
    store.resume_run(&run, &event).await.unwrap();
    assert_eq!(store.get_run(&run.id).await.unwrap(), Some(run));
    assert_eq!(store.list_events(None, None).await.unwrap(), [event]);
    store.close().await.unwrap();
}

#[tokio::test]
async fn approval_decisions_are_idempotent_immutable_and_emit_exactly_one_event() {
    for decision in [ApprovalDecision::Approved, ApprovalDecision::Denied] {
        let (store, session, run) = seeded().await;
        let approval = request(&session, &run);
        store.save_approval(&approval).await.unwrap();
        let resolved = store
            .resolve_approval(&approval.id, decision)
            .await
            .unwrap();
        assert_eq!(resolved.status, ApprovalStatus::from(decision));
        assert!(resolved.resolved_at.is_some());
        assert_eq!(
            store
                .resolve_approval(&approval.id, decision)
                .await
                .unwrap(),
            resolved
        );
        let opposite = match decision {
            ApprovalDecision::Approved => ApprovalDecision::Denied,
            ApprovalDecision::Denied => ApprovalDecision::Approved,
        };
        assert!(matches!(
            store.resolve_approval(&approval.id, opposite).await,
            Err(Error::ApprovalConflict { .. })
        ));
        let events = store.list_events(None, Some(&run.id)).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventType::ApprovalResolved);
        assert_eq!(events[0].payload, serde_json::from_value::<JsonObject>(json!({
            "approval_id":approval.id, "decision":decision, "tool_call_id":approval.tool_call.id,
        })).unwrap());
        assert!(matches!(
            store.resolve_approval("missing", decision).await,
            Err(Error::UnknownApproval(_))
        ));
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn concurrent_conflicting_decisions_have_one_winner() {
    let (store, session, run) = seeded().await;
    let approval = request(&session, &run);
    store.save_approval(&approval).await.unwrap();
    let (approved, denied) = tokio::join!(
        store.resolve_approval(&approval.id, ApprovalDecision::Approved),
        store.resolve_approval(&approval.id, ApprovalDecision::Denied),
    );
    assert!(matches!(
        (&approved, &denied),
        (Ok(_), Err(Error::ApprovalConflict { .. })) | (Err(Error::ApprovalConflict { .. }), Ok(_))
    ));
    assert_eq!(store.list_events(None, None).await.unwrap().len(), 1);
    store.close().await.unwrap();
}

#[tokio::test]
async fn approval_filters_combine_and_tied_dates_use_ids() {
    let (store, session, first_run) = seeded().await;
    let second_run = Run::new(&session.id, "agent_other");
    store.save_run(&second_run).await.unwrap();
    let mut a = request(&session, &first_run);
    a.id = "appr_a".into();
    let mut b = request(&session, &first_run);
    b.id = "appr_b".into();
    b.created_at = a.created_at;
    let mut c = request(&session, &second_run);
    c.id = "appr_c".into();
    c.created_at = a.created_at;
    for approval in [&c, &a, &b] {
        store.save_approval(approval).await.unwrap();
    }
    let b = store
        .resolve_approval(&b.id, ApprovalDecision::Approved)
        .await
        .unwrap();
    assert_eq!(
        store.list_approvals(None, None).await.unwrap(),
        [a.clone(), b.clone(), c.clone()]
    );
    assert_eq!(
        store
            .list_approvals(Some(&first_run.id), None)
            .await
            .unwrap(),
        [a.clone(), b.clone()]
    );
    assert_eq!(
        store
            .list_approvals(None, Some(ApprovalStatus::Pending))
            .await
            .unwrap(),
        [a, c]
    );
    assert_eq!(
        store
            .list_approvals(Some(&first_run.id), Some(ApprovalStatus::Approved))
            .await
            .unwrap(),
        [b]
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn event_filters_combine_without_losing_unscoped_events_or_stable_ordering() {
    let (store, first_session, first_run) = seeded().await;
    let second_session = Session::default();
    store.save_session(&second_session).await.unwrap();
    let second_run = Run::new(&second_session.id, "agent_other");
    store.save_run(&second_run).await.unwrap();
    let timestamp = now();
    let events: Vec<_> = [
        ("evt_a", None, None),
        ("evt_b", Some(&first_session.id), None),
        ("evt_c", Some(&first_session.id), Some(&first_run.id)),
        ("evt_d", Some(&second_session.id), None),
        ("evt_e", Some(&second_session.id), Some(&second_run.id)),
    ]
    .into_iter()
    .map(|(id, session, run)| Event {
        id: id.into(),
        session_id: session.cloned(),
        run_id: run.cloned(),
        created_at: timestamp,
        ..Event::new(EventType::RunStarted)
    })
    .collect();
    for event in events.iter().rev() {
        store.append_event(event).await.unwrap();
    }
    for session in [None, Some(first_session.id.as_str()), Some("' OR 1=1 --")] {
        for run in [
            None,
            Some(first_run.id.as_str()),
            Some(second_run.id.as_str()),
        ] {
            let expected: Vec<_> = events
                .iter()
                .filter(|event| {
                    session.is_none_or(|id| event.session_id.as_deref() == Some(id))
                        && run.is_none_or(|id| event.run_id.as_deref() == Some(id))
                })
                .cloned()
                .collect();
            assert_eq!(store.list_events(session, run).await.unwrap(), expected);
        }
    }
    store.close().await.unwrap();
}

#[tokio::test]
async fn file_state_survives_reopen_and_close_is_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("store.sqlite");
    let store = Store::open(&path).await.unwrap();
    let session = Session::with_title("Durable").unwrap();
    store.save_session(&session).await.unwrap();
    store.close().await.unwrap();
    store.close().await.unwrap();
    assert!(matches!(
        store.get_session(&session.id).await,
        Err(Error::StoreClosed)
    ));
    assert!(matches!(
        store.save_session(&session).await,
        Err(Error::StoreClosed)
    ));
    let reopened = Store::open(&path).await.unwrap();
    assert_eq!(
        reopened.get_session(&session.id).await.unwrap(),
        Some(session)
    );
    reopened.close().await.unwrap();
    let mut connection =
        SqliteConnection::connect_with(&SqliteConnectOptions::new().filename(path))
            .await
            .unwrap();
    let journal: String = sqlx::query_scalar("PRAGMA journal_mode")
        .fetch_one(&mut connection)
        .await
        .unwrap();
    assert_eq!(journal, "wal");
    let version: String = sqlx::query_scalar("SELECT value FROM meta WHERE key='schema_version'")
        .fetch_one(&mut connection)
        .await
        .unwrap();
    assert_eq!(version, "1");
    connection.close().await.unwrap();
}

#[tokio::test]
async fn incompatible_and_unversioned_databases_are_rejected_without_schema_changes() {
    for version in [None, Some("2"), Some("not-a-version")] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unsupported.sqlite");
        let mut connection = SqliteConnection::connect_with(
            &SqliteConnectOptions::new()
                .filename(&path)
                .create_if_missing(true),
        )
        .await
        .unwrap();
        sqlx::query("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            .execute(&mut connection)
            .await
            .unwrap();
        if let Some(version) = version {
            sqlx::query("INSERT INTO meta VALUES ('schema_version', ?)")
                .bind(version)
                .execute(&mut connection)
                .await
                .unwrap();
        }
        connection.close().await.unwrap();
        let result = Store::open(&path).await;
        assert!(matches!(
            result,
            Err(Error::UnsupportedSchema { .. }) | Err(Error::UnversionedDatabase)
        ));
        let mut connection =
            SqliteConnection::connect_with(&SqliteConnectOptions::new().filename(&path))
                .await
                .unwrap();
        let tables: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                .fetch_one(&mut connection)
                .await
                .unwrap();
        assert_eq!(tables, 1);
        let journal: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&mut connection)
            .await
            .unwrap();
        assert_eq!(journal, "delete");
        connection.close().await.unwrap();
    }
}
