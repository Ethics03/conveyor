use std::time::Duration;

use conveyor_core::{CancellationToken, Cancelled, DEFAULT_CANCELLATION_REASON};

#[test]
fn the_first_accepted_reason_wins_and_blank_reasons_use_the_default() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
    assert_eq!(token.reason(), None);
    assert_eq!(token.check(), Ok(()));

    assert!(token.cancel("  \n"));
    assert!(!token.cancel("Shutting down"));
    assert!(token.is_cancelled());
    assert_eq!(token.reason(), Some(DEFAULT_CANCELLATION_REASON));
    assert_eq!(
        token.check(),
        Err(Cancelled::new(DEFAULT_CANCELLATION_REASON))
    );
}

#[test]
fn reasons_are_trimmed_and_shared_by_clones() {
    let token = CancellationToken::new();
    let clone = token.clone();
    assert!(clone.cancel("  Shutting down \n"));
    assert_eq!(token.reason(), Some("Shutting down"));
    assert_eq!(token.check().unwrap_err().to_string(), "Shutting down");
}

#[tokio::test]
async fn waiting_resolves_immediately_for_an_already_cancelled_token() {
    let token = CancellationToken::new();
    token.cancel("Done");
    assert_eq!(token.cancelled().await.reason(), "Done");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_waiter_wakes_with_the_accepted_reason() {
    let token = CancellationToken::new();
    let waiters: Vec<_> = (0..8)
        .map(|_| {
            let token = token.clone();
            tokio::spawn(async move { token.cancelled().await })
        })
        .collect();
    tokio::task::yield_now().await;

    assert!(token.cancel("Interrupted"));
    for waiter in waiters {
        let cancelled = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("waiter woke")
            .expect("waiter task completed");
        assert_eq!(cancelled.reason(), "Interrupted");
    }
}

#[tokio::test]
async fn dropping_a_wait_leaves_the_token_usable() {
    let token = CancellationToken::new();
    let pending = tokio::time::timeout(Duration::from_millis(5), token.cancelled()).await;
    assert!(pending.is_err());
    assert!(!token.is_cancelled());

    token.cancel("Later");
    assert_eq!(token.cancelled().await.reason(), "Later");
}
