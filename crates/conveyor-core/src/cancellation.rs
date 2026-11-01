//! Per-turn cancellation signals and the first accepted cancellation reason.

use std::{
    pin::pin,
    sync::{Arc, OnceLock},
};

use tokio::sync::Notify;

/// The reason recorded when cancellation is requested without one.
pub const DEFAULT_CANCELLATION_REASON: &str = "Run cancelled by user";

/// A cloneable, cooperative cancellation signal shared by one turn's work.
///
/// The first accepted [`cancel`](Self::cancel) call wins and fixes the reason;
/// later calls are ignored. Waiting needs no particular executor.
///
/// ```
/// use conveyor_core::CancellationToken;
///
/// let token = CancellationToken::new();
/// let observer = token.clone();
/// assert!(token.cancel("Shutting down"));
/// assert!(!token.cancel("Too late"));
/// assert_eq!(observer.check().unwrap_err().reason(), "Shutting down");
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    reason: OnceLock<String>,
    notify: Notify,
}

impl CancellationToken {
    /// Create a token that has not been cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation, returning whether this call was the first accepted one.
    ///
    /// A blank reason is replaced by [`DEFAULT_CANCELLATION_REASON`].
    pub fn cancel(&self, reason: &str) -> bool {
        let reason = match reason.trim() {
            "" => DEFAULT_CANCELLATION_REASON,
            reason => reason,
        };
        let accepted = self.inner.reason.set(reason.to_owned()).is_ok();
        if accepted {
            self.inner.notify.notify_waiters();
        }
        accepted
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.reason.get().is_some()
    }

    /// The accepted cancellation reason, if cancellation has been requested.
    pub fn reason(&self) -> Option<&str> {
        self.inner.reason.get().map(String::as_str)
    }

    /// Return [`Cancelled`] at a safe boundary once cancellation is requested.
    pub fn check(&self) -> Result<(), Cancelled> {
        match self.reason() {
            Some(reason) => Err(Cancelled::new(reason)),
            None => Ok(()),
        }
    }

    /// Wait until cancellation is requested, then return its reason.
    ///
    /// Resolves immediately for an already-cancelled token. Dropping the future
    /// stops waiting without affecting the token.
    pub async fn cancelled(&self) -> Cancelled {
        loop {
            // Register before checking so a concurrent `cancel` cannot be missed.
            let mut notified = pin!(self.inner.notify.notified());
            notified.as_mut().enable();
            if let Some(reason) = self.reason() {
                return Cancelled::new(reason);
            }
            notified.await;
        }
    }
}

/// Work stopped at a safe boundary because its turn was cancelled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct Cancelled {
    reason: String,
}

impl Cancelled {
    /// Record cancellation with the given reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// The accepted cancellation reason.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}
