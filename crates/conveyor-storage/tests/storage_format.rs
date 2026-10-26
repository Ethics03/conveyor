use std::path::Path;

use conveyor_core::models::*;
use conveyor_storage::Store;
use serde::Deserialize;
use sqlx::{Connection, SqliteConnection, sqlite::SqliteConnectOptions};

#[derive(Debug, PartialEq, Deserialize)]
struct Records {
    session: Session,
    runs: Vec<Run>,
    messages: Vec<Message>,
    events: Vec<Event>,
    approvals: Vec<ApprovalRequest>,
}

#[derive(Deserialize)]
struct Fixture {
    records: Records,
}

async fn snapshot(store: &Store, id: &str) -> Records {
    Records {
        session: store.get_session(id).await.unwrap().unwrap(),
        runs: store.list_runs(id).await.unwrap(),
        messages: store.list_messages(id).await.unwrap(),
        events: store.list_events(Some(id), None).await.unwrap(),
        approvals: store.list_approvals(None, None).await.unwrap(),
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../test-support/fixtures/storage-v1/records.json"
    ))
    .unwrap()
}

async fn restore_fixture(path: &Path) -> Store {
    let mut connection = SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(false),
    )
    .await
    .unwrap();
    // The trusted fixture defines tables alphabetically. Disable foreign keys
    // only during restoration, before opening the production Store.
    sqlx::raw_sql(include_str!(
        "../../../test-support/fixtures/storage-v1/database.sql"
    ))
    .execute(&mut connection)
    .await
    .unwrap();
    connection.close().await.unwrap();
    Store::open(path).await.unwrap()
}

#[tokio::test]
async fn reads_version_one_records_without_changing_offsets_or_opaque_metadata() {
    let directory = tempfile::tempdir().unwrap();
    let store = restore_fixture(&directory.path().join("format.sqlite")).await;
    let fixture = fixture();
    let loaded = snapshot(&store, &fixture.records.session.id).await;
    assert_eq!(loaded, fixture.records);
    assert_eq!(
        loaded.session.updated_at.offset().local_minus_utc(),
        5 * 3600 + 30 * 60
    );
    assert_eq!(
        loaded.messages[1].metadata["provider_replay_state"]["items"][0]["encrypted_content"],
        "opaque-fixture"
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn updates_version_one_records_and_reopens_all_related_state() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("roundtrip.sqlite");
    let store = restore_fixture(&path).await;
    let fixture = fixture();
    let session = &fixture.records.session;
    store
        .set_session_title(&session.id, "Durable runtime 🦀")
        .await
        .unwrap();
    store
        .resolve_approval(&fixture.records.approvals[0].id, ApprovalDecision::Approved)
        .await
        .unwrap();
    let mut resumed = fixture.records.runs[1].clone();
    resumed.status = RunStatus::Running;
    let mut event = Event::new(EventType::RunResumed);
    event.session_id = Some(session.id.clone());
    event.run_id = Some(resumed.id.clone());
    store.resume_run(&resumed, &event).await.unwrap();
    let run = Run::new(&session.id, "agent_new");
    store.save_run(&run).await.unwrap();
    let mut message = Message::new(&session.id, MessageRole::Assistant, "Persisted: café 🦀");
    message.run_id = Some(run.id.clone());
    message
        .tool_calls
        .push(ToolCall::new("read_file", JsonObject::new()));
    store.save_message(&message).await.unwrap();
    let expected = snapshot(&store, &session.id).await;
    assert_eq!(expected.session.title_source, SessionTitleSource::User);
    assert_eq!(expected.approvals[0].status, ApprovalStatus::Approved);
    assert_eq!(
        expected
            .runs
            .iter()
            .find(|run| run.id == resumed.id)
            .unwrap()
            .status,
        RunStatus::Running
    );
    assert_eq!(expected.messages.len(), fixture.records.messages.len() + 1);
    assert_eq!(expected.events.len(), fixture.records.events.len() + 2);
    store.close().await.unwrap();
    let reopened = Store::open(&path).await.unwrap();
    assert_eq!(snapshot(&reopened, &session.id).await, expected);
    reopened.close().await.unwrap();
}
