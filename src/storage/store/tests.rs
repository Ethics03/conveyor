use std::{sync::Arc, time::Duration};

use tokio::sync::oneshot;

use super::*;

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
