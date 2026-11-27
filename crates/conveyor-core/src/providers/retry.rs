//! Timeouts, retries, and error responses shared by the HTTP provider
//! adapters.

use std::time::Duration;

use conveyor_http::{Bytes, HeaderMap, HttpTransport, Request, Response, StatusCode, Uri};
use serde_json::Value;

use super::{BoxError, ProviderError};
use crate::Error;

const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(500);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(8);
/// The longest server-requested wait honored; longer waits fall back to
/// the usual backoff, so a call does not stall for minutes.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
/// The most characters of an unparsed error body kept in a message.
const ERROR_BODY_LIMIT: usize = 1_000;

/// Why one attempt failed, and whether and when to try again.
pub(crate) struct Failure {
    pub(crate) error: ProviderError,
    pub(crate) retryable: bool,
    pub(crate) retry_after: Option<Duration>,
}

/// How long each attempt may take and how many times a failed one is
/// repeated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryPolicy {
    /// The deadline for each attempt, including reading the body.
    pub(crate) timeout: Duration,
    pub(crate) max_retries: u32,
}

/// Send the request `build` makes until a response succeeds, a failure is
/// final, or the retries are spent.
///
/// Transport failures are retried when the transport says they may succeed,
/// timeouts always, and unsuccessful responses as `api_failure` judges them.
/// Retries wait for the server's requested delay or back off exponentially.
pub(crate) async fn send(
    transport: &dyn HttpTransport,
    policy: RetryPolicy,
    build: impl Fn() -> Result<Request<Bytes>, ProviderError>,
    api_failure: impl Fn(&Response<Bytes>) -> Failure,
) -> Result<Response<Bytes>, ProviderError> {
    let mut attempt = 0;
    loop {
        let outcome = tokio::time::timeout(policy.timeout, transport.send(build()?)).await;
        let failure = match outcome {
            Ok(Ok(response)) if response.status().is_success() => return Ok(response),
            Ok(Ok(response)) => api_failure(&response),
            Ok(Err(error)) => Failure {
                retryable: error.is_retryable(),
                retry_after: None,
                error: ProviderError::Transport(BoxError::from(error)),
            },
            Err(_) => Failure {
                retryable: true,
                retry_after: None,
                error: ProviderError::Timeout(policy.timeout),
            },
        };
        if !failure.retryable || attempt >= policy.max_retries {
            return Err(failure.error);
        }
        tokio::time::sleep(failure.retry_after.unwrap_or_else(|| backoff(attempt))).await;
        attempt += 1;
    }
}

/// The endpoint `path` below `base_url`, which must be an absolute URL.
pub(crate) fn endpoint(base_url: &str, path: &str) -> crate::Result<Uri> {
    format!("{}{path}", base_url.trim_end_matches('/'))
        .parse::<Uri>()
        .ok()
        .filter(|uri| uri.scheme().is_some() && uri.authority().is_some())
        .ok_or(Error::InvalidBaseUrl)
}

/// Honor `x-should-retry`, otherwise retry timeouts, conflicts, rate limits,
/// and server errors.
pub(crate) fn should_retry(status: StatusCode, headers: &HeaderMap) -> bool {
    match header_str(headers, "x-should-retry") {
        Some("true") => true,
        Some("false") => false,
        _ => matches!(status.as_u16(), 408 | 409 | 429) || status.is_server_error(),
    }
}

/// A server-requested delay of at most a minute, from `retry-after-ms` or
/// `retry-after` in seconds.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let millis = header_str(headers, "retry-after-ms")
        .and_then(|value| value.trim().parse::<f64>().ok())
        .map(|millis| millis / 1_000.0);
    let seconds = millis.or_else(|| {
        header_str(headers, "retry-after").and_then(|value| value.trim().parse::<f64>().ok())
    })?;
    honored(seconds)
}

/// A delay an error message asks for, such as "Please try again in 1.5s" or
/// "try again in 6m0s", when it is at most a minute.
#[cfg(feature = "openai")]
pub(crate) fn hinted_retry(message: &str) -> Option<Duration> {
    const MARKER: &str = "try again in ";
    let lower = message.to_ascii_lowercase();
    let mut rest = &lower[lower.find(MARKER)? + MARKER.len()..];
    let mut seconds = None;
    // Each part is a number and a unit, such as `6m` then `0s`; anything
    // else, such as the full stop ending a sentence, ends the duration.
    while rest.starts_with(|character: char| character.is_ascii_digit()) {
        let end = rest
            .find(|character: char| !(character.is_ascii_digit() || character == '.'))
            .unwrap_or(rest.len());
        let amount: f64 = rest[..end].parse().ok()?;
        let unit = &rest[end..];
        let unit_end = unit
            .find(|character: char| !character.is_ascii_alphabetic())
            .unwrap_or(unit.len());
        let scale = match &unit[..unit_end] {
            "ms" => 0.001,
            "s" => 1.0,
            "m" => 60.0,
            "h" => 3_600.0,
            _ => return None,
        };
        seconds = Some(seconds.unwrap_or(0.0) + amount * scale);
        rest = &unit[unit_end..];
    }
    honored(seconds?)
}

fn honored(seconds: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|delay| !delay.is_zero() && *delay <= MAX_RETRY_AFTER)
}

/// Exponential backoff from 0.5 seconds, capped at 8 seconds.
fn backoff(attempt: u32) -> Duration {
    INITIAL_RETRY_DELAY
        .saturating_mul(2u32.saturating_pow(attempt))
        .min(MAX_RETRY_DELAY)
}

pub(crate) fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// A message for an error response whose body could not be parsed: the
/// start of the body, or the status's reason when the body is empty.
pub(crate) fn error_text(response: &Response<Bytes>) -> String {
    let body = String::from_utf8_lossy(response.body());
    let body = body.trim();
    match body.char_indices().nth(ERROR_BODY_LIMIT) {
        Some((end, _)) => format!("{}…", &body[..end]),
        None if body.is_empty() => response
            .status()
            .canonical_reason()
            .unwrap_or("")
            .to_owned(),
        None => body.to_owned(),
    }
}

/// Drop null object fields recursively.
pub(crate) fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key, without_nulls(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

#[cfg(all(test, feature = "openai"))]
mod tests {
    use std::time::Duration;

    use super::hinted_retry;

    #[test]
    fn wait_hints_in_messages_are_read() {
        let hint = |message| hinted_retry(message);
        assert_eq!(
            hint("Rate limit reached. Please try again in 1.5s. Visit ..."),
            Some(Duration::from_millis(1_500))
        );
        assert_eq!(hint("try again in 250ms"), Some(Duration::from_millis(250)));
        assert_eq!(hint("Try again in 0m20s."), Some(Duration::from_secs(20)));
        assert_eq!(hint("try again in 6m0s"), None, "longer than a minute");
        assert_eq!(hint("try again in a bit"), None);
        assert_eq!(hint("slow down"), None);
    }
}
