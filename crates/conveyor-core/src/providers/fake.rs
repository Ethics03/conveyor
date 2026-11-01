//! A scripted, deterministic provider for tests and offline examples.

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard, PoisonError},
};

use super::{ModelLimits, Provider, ProviderError, ProviderRequest, ProviderResponse};
use crate::BoxFuture;

/// Limits reported by [`FakeProvider`] unless configured otherwise.
pub const DEFAULT_FAKE_MODEL_LIMITS: ModelLimits = match ModelLimits::new(200_000, 4_096) {
    Ok(limits) => limits,
    Err(_) => panic!("default fake model limits are valid"),
};

type Step = Result<ProviderResponse, ProviderError>;

/// Returns scripted responses in order, then `"ok"` once the script is exhausted.
///
/// Every request is recorded for inspection.
///
/// ```
/// use conveyor_core::providers::{Provider, ProviderMessage, ProviderRequest, fake::FakeProvider};
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), conveyor_core::providers::ProviderError> {
/// let provider = FakeProvider::new(["hello"]);
/// let request = ProviderRequest::new(vec![ProviderMessage::user("hi")]);
/// assert_eq!(provider.generate(request.clone()).await?.content, "hello");
/// assert_eq!(provider.generate(request).await?.content, "ok");
/// assert_eq!(provider.requests().len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct FakeProvider {
    steps: Mutex<VecDeque<Step>>,
    requests: Mutex<Vec<ProviderRequest>>,
    limits: ModelLimits,
}

impl FakeProvider {
    /// Script successful responses; strings become plain text replies.
    pub fn new<I>(responses: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ProviderResponse>,
    {
        Self::with_results(responses.into_iter().map(|response| Ok(response.into())))
    }

    /// Script responses and failures, returned in order.
    pub fn with_results(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            steps: Mutex::new(steps.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            limits: DEFAULT_FAKE_MODEL_LIMITS,
        }
    }

    /// Report `limits` for every model.
    pub fn with_model_limits(mut self, limits: ModelLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The requests received so far, in order.
    pub fn requests(&self) -> Vec<ProviderRequest> {
        lock(&self.requests).clone()
    }
}

impl Default for FakeProvider {
    /// A provider that always answers `"ok"`.
    fn default() -> Self {
        Self::with_results([])
    }
}

impl Provider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }

    fn model_limits(&self, _model: Option<&str>) -> ModelLimits {
        self.limits
    }

    fn generate(&self, request: ProviderRequest) -> BoxFuture<'_, Step> {
        // Record and consume a step only when polled, so a request dropped
        // before it starts leaves the script untouched.
        Box::pin(async move {
            lock(&self.requests).push(request);
            lock(&self.steps)
                .pop_front()
                .unwrap_or_else(|| Ok(ProviderResponse::message("ok")))
        })
    }
}

// Each critical section is a single push or pop, so a poisoned lock still
// guards consistent data.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
