use std::{sync::Arc, time::Duration};

use sqlx::{Execute, QueryBuilder, Row, Sqlite, query::Query, sqlite::SqliteArguments};
use tokio::sync::oneshot;

use super::*;

#[tokio::test]
async fn scoped_list_queries_search_existing_indexes() {
    let mut connection = SqliteConnection::connect("sqlite::memory:").await.unwrap();
    sqlx::raw_sql(SCHEMA)
        .execute(&mut connection)
        .await
        .unwrap();
    for query in [
        queries::events(Some("session"), None),
        queries::events(None, Some("run")),
        queries::events(Some("session"), Some("run")),
    ] {
        assert_index_search(
            &mut connection,
            query,
            "SEARCH events USING INDEX idx_events_",
        )
        .await;
    }
    for status in [None, Some(ApprovalStatus::Pending)] {
        assert_index_search(
            &mut connection,
            queries::approvals(Some("run"), status),
            "SEARCH approvals USING INDEX idx_approvals_run",
        )
        .await;
    }
    connection.close().await.unwrap();
}

async fn assert_index_search(
    connection: &mut SqliteConnection,
    mut query: Query<'_, Sqlite, SqliteArguments>,
    expected: &str,
) {
    let arguments = query.take_arguments().unwrap().unwrap();
    // Explain the production query with its real bound arguments. The SQL comes
    // only from our static list statements, never from caller-provided text.
    let mut explain = QueryBuilder::<Sqlite>::with_arguments(
        format!("EXPLAIN QUERY PLAN {}", query.sql().as_str()),
        arguments,
    );
    let plan: Vec<String> = explain
        .build()
        .fetch_all(connection)
        .await
        .unwrap()
        .iter()
        .map(|row| row.get("detail"))
        .collect();
    assert!(
        plan.iter().any(|detail| detail.starts_with(expected)),
        "{plan:?}"
    );
}

#[tokio::test]
async fn cancelling_a_backpressured_write_submits_nothing_and_close_drains_the_queue() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("backpressure.sqlite");
        let store = Arc::new(Store::open(&path).await.unwrap());
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let worker_store = Arc::clone(&store);
        let blocked_worker = tokio::spawn(async move {
            worker_store
                .call(move |_| {
                    Box::pin(async move {
                        started_tx.send(()).unwrap();
                        release_rx.await.unwrap();
                        Ok(())
                    })
                })
                .await
        });
        started_rx.await.unwrap();
        let mut writers = tokio::task::JoinSet::new();
        for _ in 0..QUEUE_CAPACITY {
            let store = Arc::clone(&store);
            writers
                .spawn(async move { store.append_event(&Event::new(EventType::RunStarted)).await });
        }
        while store.sender.lock().unwrap().as_ref().unwrap().capacity() != 0 {
            tokio::task::yield_now().await;
        }
        let cancelled = Event::new(EventType::RunCancelled);
        {
            let pending = store.append_event(&cancelled);
            tokio::pin!(pending);
            // Poll the write against a full queue, then drop it while it is
            // waiting for admission. Its event must never reach the worker.
            tokio::select! {
                biased;
                result = &mut pending => panic!("full queue admitted a write: {result:?}"),
                () = std::future::ready(()) => {}
            }
        }
        let closing_store = Arc::clone(&store);
        let closing = tokio::spawn(async move { closing_store.close().await });
        while store.sender.lock().unwrap().is_some() {
            tokio::task::yield_now().await;
        }
        assert!(!closing.is_finished());
        assert!(matches!(
            store.append_event(&cancelled).await,
            Err(Error::StoreClosed)
        ));
        release_tx.send(()).unwrap();
        blocked_worker.await.unwrap().unwrap();
        while let Some(result) = writers.join_next().await {
            result.unwrap().unwrap();
        }
        closing.await.unwrap().unwrap();
        let reopened = Store::open(&path).await.unwrap();
        let events = reopened.list_events(None, None).await.unwrap();
        assert_eq!(events.len(), QUEUE_CAPACITY);
        assert!(
            events
                .iter()
                .all(|event| event.kind == EventType::RunStarted)
        );
        reopened.close().await.unwrap();
    })
    .await
    .expect("backpressure cancellation or shutdown failed to settle");
}

#[tokio::test]
async fn accepted_transactions_finish_after_the_caller_is_cancelled() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for fail in [false, true] {
            let store = Arc::new(Store::open(":memory:").await.unwrap());
            let event = Event::new(EventType::RunCancelled);
            let expected = event.clone();
            let (started_tx, started_rx) = oneshot::channel();
            let (release_tx, release_rx) = oneshot::channel();
            let caller_store = Arc::clone(&store);
            let caller = tokio::spawn(async move {
                caller_store
                    .write(move |connection| {
                        Box::pin(async move {
                            insert_event(connection, &event).await?;
                            started_tx.send(()).unwrap();
                            release_rx.await.unwrap();
                            if fail {
                                return Err(Error::InvalidNewApproval);
                            }
                            Ok(())
                        })
                    })
                    .await
            });
            started_rx.await.unwrap();
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            release_tx.send(()).unwrap();
            // This read queues behind the submitted transaction, so it must see
            // either the committed event or the completed rollback.
            let events = store.list_events(None, None).await.unwrap();
            if fail {
                assert!(events.is_empty());
            } else {
                assert_eq!(events, [expected]);
            }
            store.close().await.unwrap();
        }
    })
    .await
    .expect("worker failed to settle a cancelled caller's transaction");
}

#[tokio::test]
async fn cancelling_close_keeps_the_worker_available_for_a_later_drain() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let store = Arc::new(Store::open(":memory:").await.unwrap());
        let event = Event::new(EventType::RunCancelled);
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let caller_store = Arc::clone(&store);
        let caller = tokio::spawn(async move {
            caller_store
                .write(move |connection| {
                    Box::pin(async move {
                        insert_event(connection, &event).await?;
                        started_tx.send(()).unwrap();
                        release_rx.await.unwrap();
                        Ok(())
                    })
                })
                .await
        });
        started_rx.await.unwrap();
        let close_store = Arc::clone(&store);
        let closing = tokio::spawn(async move { close_store.close().await });
        while store.sender.lock().unwrap().is_some() {
            tokio::task::yield_now().await;
        }
        assert!(!closing.is_finished());
        assert!(matches!(
            store.list_sessions(None).await,
            Err(Error::StoreClosed)
        ));
        closing.abort();
        assert!(closing.await.unwrap_err().is_cancelled());
        assert!(store.worker.lock().await.is_some());
        release_tx.send(()).unwrap();
        caller.await.unwrap().unwrap();
        store.close().await.unwrap();
        store.close().await.unwrap();
        assert!(store.worker.lock().await.is_none());
    })
    .await
    .expect("closing lost the worker or failed to drain accepted work");
}
