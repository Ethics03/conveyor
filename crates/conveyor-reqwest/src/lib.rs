//! A reqwest backend for Conveyor's async HTTP transport.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use conveyor_http::HttpTransport;
//! use conveyor_reqwest::ReqwestTransport;
//!
//! # fn example() -> Result<(), conveyor_http::Error> {
//! // Build one transport and share it, so every request reuses its connections.
//! let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
//! # let _ = transport;
//! # Ok(())
//! # }
//! ```

use std::{fmt, time::Duration};

use conveyor_http::{BoxFuture, Bytes, Error, ErrorKind, HttpTransport, Request, Response};

/// How long [`ReqwestTransport::new`] waits to connect before the attempt fails.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a connection may sit idle before the OS sends keepalive probes.
/// A model call can be silent for minutes; probes stop NATs and proxies from
/// dropping the connection meanwhile.
pub const TCP_KEEPALIVE: Duration = Duration::from_secs(30);
/// The largest successful response body read by default; larger bodies fail.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// The most of an unsuccessful response's body that is kept.
pub const ERROR_BODY_BYTES: usize = 64 * 1024;

/// Sends Conveyor's HTTP requests with reqwest.
///
/// Build one and share it, for example as `Arc<dyn HttpTransport>`, so that
/// requests reuse its connection pool. Cloning is cheap and shares the pool.
///
/// A transport from [`new`](Self::new) never follows redirects: a 3xx
/// response is returned as it is, so the credentials sent to one host are
/// never forwarded to another. It sets no overall deadline; callers bound
/// each attempt, as providers do with their own timeout.
///
/// Response bodies are read whole. An unsuccessful response keeps at most
/// [`ERROR_BODY_BYTES`] of its body, enough for the error it describes; a
/// successful one larger than the body limit fails with
/// [`ErrorKind::Body`].
///
/// A connection that fails after the request is sent, before the whole
/// response arrives, is [`ErrorKind::Interrupted`], which is retryable. This
/// covers a pooled connection the server closed while it sat idle.
#[derive(Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
    max_body_bytes: usize,
}

impl ReqwestTransport {
    /// A transport using Rustls with the system's certificate verifier,
    /// HTTP/2 where the server offers it, the system's proxy settings, a
    /// [`DEFAULT_CONNECT_TIMEOUT`], [`TCP_KEEPALIVE`] probes, and no
    /// redirects. It sends no `User-Agent` of its own; a provider that needs
    /// one sets it on its requests.
    ///
    /// Fails with [`ErrorKind::Request`] when the TLS backend or proxy
    /// settings cannot be initialized.
    pub fn new() -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .tcp_keepalive(TCP_KEEPALIVE)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| Error::new(ErrorKind::Request, error))?;
        Ok(Self::from(client))
    }

    /// Fail successful responses whose body is larger than `bytes`.
    #[must_use]
    pub fn with_max_body_bytes(mut self, bytes: usize) -> Self {
        self.max_body_bytes = bytes;
        self
    }

    /// The underlying reqwest client.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

impl From<reqwest::Client> for ReqwestTransport {
    /// A transport over a client configured by the caller, which keeps that
    /// client's own redirect, timeout, TLS, and proxy policy.
    fn from(client: reqwest::Client) -> Self {
        Self {
            client,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }
}

impl fmt::Debug for ReqwestTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReqwestTransport")
            .field("max_body_bytes", &self.max_body_bytes)
            .finish_non_exhaustive()
    }
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: Request<Bytes>) -> BoxFuture<'_, Result<Response<Bytes>, Error>> {
        Box::pin(async move {
            let request = reqwest::Request::try_from(request)
                .map_err(|error| Error::new(ErrorKind::Request, error))?;
            let mut response = self.client.execute(request).await.map_err(classify)?;

            let status = response.status();
            let version = response.version();
            let headers = std::mem::take(response.headers_mut());
            let body = if status.is_success() {
                read_body(&mut response, self.max_body_bytes).await?
            } else {
                read_prefix(&mut response, ERROR_BODY_BYTES).await?
            };

            let mut reply = Response::new(body);
            *reply.status_mut() = status;
            *reply.version_mut() = version;
            *reply.headers_mut() = headers;
            Ok(reply)
        })
    }
}

/// The whole body, failing once it exceeds `limit` bytes.
async fn read_body(response: &mut reqwest::Response, limit: usize) -> Result<Bytes, Error> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(Error::new(ErrorKind::Body, BodyTooLarge(limit)));
    }
    let mut body = Vec::new();
    while let Some(chunk) = next_chunk(response).await? {
        if chunk.len() > limit - body.len() {
            return Err(Error::new(ErrorKind::Body, BodyTooLarge(limit)));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// At most the first `limit` bytes of the body; the rest is not read.
async fn read_prefix(response: &mut reqwest::Response, limit: usize) -> Result<Bytes, Error> {
    let mut body = Vec::new();
    while let Some(chunk) = next_chunk(response).await? {
        let room = limit - body.len();
        if chunk.len() >= room {
            body.extend_from_slice(&chunk[..room]);
            break;
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// The next body chunk. A failed read means the connection broke before the
/// response was complete.
async fn next_chunk(response: &mut reqwest::Response) -> Result<Option<Bytes>, Error> {
    response
        .chunk()
        .await
        .map_err(|error| Error::new(ErrorKind::Interrupted, error))
}

/// The transport error kind for a failure to send a request and receive its
/// response head. A request reqwest could not build is the only one that
/// would fail again unchanged; anything else after connecting, such as a
/// connection closed before the response or an HTTP/2 stream reset, is an
/// interruption.
fn classify(error: reqwest::Error) -> Error {
    let kind = if error.is_timeout() {
        ErrorKind::Timeout
    } else if error.is_connect() {
        ErrorKind::Connect
    } else if error.is_builder() {
        ErrorKind::Request
    } else {
        ErrorKind::Interrupted
    };
    Error::new(kind, error)
}

/// A successful response body exceeded the transport's limit.
#[derive(Debug)]
struct BodyTooLarge(usize);

impl fmt::Display for BodyTooLarge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "response body exceeds {} bytes", self.0)
    }
}

impl std::error::Error for BodyTooLarge {}
