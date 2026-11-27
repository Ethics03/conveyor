//! Async HTTP transport contracts for Conveyor's provider adapters.
//!
//! Callers build [`Request`]s and interpret [`Response`]s; an [`HttpTransport`]
//! only moves bytes.

use std::{error::Error as StdError, fmt, future::Future, pin::Pin};

pub use bytes::Bytes;
pub use http::{
    HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri, header,
};

/// A boxed error from a transport implementation.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// A boxed `Send` future returned by object-safe transport methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Sends complete HTTP requests and reads complete responses.
///
/// A transport returns every received response, including error statuses;
/// interpreting status codes and retrying belongs to the caller. Dropping the
/// returned future must abort the request.
pub trait HttpTransport: fmt::Debug + Send + Sync {
    /// Send `request` and read the whole response body.
    fn send(&self, request: Request<Bytes>) -> BoxFuture<'_, Result<Response<Bytes>, Error>>;
}

/// Where an HTTP exchange failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// No connection could be established.
    Connect,
    /// The transport's own deadline elapsed.
    Timeout,
    /// The request could not be sent as built, such as a relative URL.
    Request,
    /// The connection failed after the request was sent, before the whole
    /// response arrived. The server may already have acted on the request.
    Interrupted,
    /// The response body was rejected, such as for exceeding a size limit.
    Body,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Connect => "connection",
            Self::Timeout => "timeout",
            Self::Request => "request",
            Self::Interrupted => "exchange",
            Self::Body => "response body",
        })
    }
}

/// An HTTP exchange that produced no response.
#[derive(Debug, thiserror::Error)]
#[error("HTTP {kind} failed")]
pub struct Error {
    kind: ErrorKind,
    #[source]
    source: BoxError,
}

impl Error {
    pub fn new(kind: ErrorKind, source: impl Into<BoxError>) -> Self {
        Self {
            kind,
            source: source.into(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Whether sending the same request again may succeed: the connection
    /// could not be made, timed out, or was interrupted.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::Connect | ErrorKind::Timeout | ErrorKind::Interrupted
        )
    }
}
