use std::time::Duration;

use conveyor_http::{Bytes, ErrorKind, HttpTransport, Method, Request, StatusCode};
use conveyor_reqwest::{ERROR_BODY_BYTES, ReqwestTransport};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
    time::timeout,
};

const WAIT: Duration = Duration::from_secs(5);

/// One request as the server read it.
#[derive(Debug)]
struct Received {
    head: String,
    body: Vec<u8>,
}

impl Received {
    fn request_line(&self) -> &str {
        self.head.lines().next().unwrap_or_default()
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }
}

/// Serves one connection per scripted response, closing each after replying.
async fn serve(responses: Vec<Vec<u8>>) -> (String, JoinHandle<Vec<Received>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let mut received = Vec::new();
        for response in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = Vec::new();
            let head_end = loop {
                let mut chunk = [0; 4096];
                let read = stream.read(&mut chunk).await.unwrap();
                assert!(read > 0, "client closed before sending a request");
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let head = String::from_utf8(buffer[..head_end].to_vec()).unwrap();
            let length = head
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            let mut body = buffer[head_end..].to_vec();
            while body.len() < length {
                let mut chunk = [0; 4096];
                let read = stream.read(&mut chunk).await.unwrap();
                body.extend_from_slice(&chunk[..read]);
            }
            received.push(Received { head, body });
            stream.write_all(&response).await.unwrap();
            stream.shutdown().await.unwrap();
        }
        received
    });
    (base, handle)
}

/// A raw HTTP/1.1 response that closes its connection.
fn response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut raw = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        raw.push_str(&format!("{name}: {value}\r\n"));
    }
    raw.push_str("\r\n");
    let mut raw = raw.into_bytes();
    raw.extend_from_slice(body);
    raw
}

fn post(url: &str, body: &'static str) -> Request<Bytes> {
    Request::builder()
        .method(Method::POST)
        .uri(url)
        .header("content-type", "application/json")
        .header("x-api-key", "secret")
        .body(Bytes::from_static(body.as_bytes()))
        .unwrap()
}

#[tokio::test]
async fn requests_and_responses_round_trip() {
    let (base, server) = serve(vec![response(
        "201 Created",
        &[("x-request-id", "req_1")],
        br#"{"ok":true}"#,
    )])
    .await;
    let transport = ReqwestTransport::new().unwrap();

    let reply = timeout(
        WAIT,
        transport.send(post(&format!("{base}/v1/responses"), r#"{"model":"m"}"#)),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(reply.status(), StatusCode::CREATED);
    assert_eq!(reply.headers()["x-request-id"], "req_1");
    assert_eq!(reply.body().as_ref(), br#"{"ok":true}"#);
    let received = server.await.unwrap();
    assert_eq!(received[0].request_line(), "POST /v1/responses HTTP/1.1");
    assert_eq!(received[0].header("x-api-key"), Some("secret"));
    assert_eq!(received[0].header("content-type"), Some("application/json"));
    assert_eq!(received[0].body, br#"{"model":"m"}"#);
}

#[tokio::test]
async fn unsuccessful_responses_are_returned_with_a_capped_body() {
    let large = vec![b'e'; ERROR_BODY_BYTES * 3];
    let (base, server) = serve(vec![
        response(
            "429 Too Many Requests",
            &[("retry-after", "2")],
            br#"{"error":{}}"#,
        ),
        response("503 Service Unavailable", &[], &large),
    ])
    .await;
    let transport = ReqwestTransport::new().unwrap();

    let limited = transport.send(post(&base, "{}")).await.unwrap();
    assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers()["retry-after"], "2");
    assert_eq!(limited.body().as_ref(), br#"{"error":{}}"#);

    let unavailable = transport.send(post(&base, "{}")).await.unwrap();
    assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable.body().len(), ERROR_BODY_BYTES);
    server.await.unwrap();
}

#[tokio::test]
async fn redirects_are_returned_rather_than_followed() {
    // A second server stands in for where a redirect would send the key.
    let elsewhere = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = format!("http://{}/collect", elsewhere.local_addr().unwrap());
    let (base, server) = serve(vec![response(
        "307 Temporary Redirect",
        &[("location", &target)],
        b"",
    )])
    .await;
    let transport = ReqwestTransport::new().unwrap();

    let reply = transport.send(post(&base, "{}")).await.unwrap();

    assert_eq!(reply.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(reply.headers()["location"], target.as_str());
    server.await.unwrap();
    let followed = timeout(Duration::from_millis(300), elsewhere.accept()).await;
    assert!(followed.is_err(), "the redirect target was contacted");
}

#[tokio::test]
async fn successful_bodies_over_the_limit_fail() {
    let body = vec![b'x'; 100];
    let mut streamed = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
    streamed.extend_from_slice(&body);
    let (base, server) = serve(vec![response("200 OK", &[], &body), streamed]).await;
    let transport = ReqwestTransport::new().unwrap().with_max_body_bytes(10);

    // Refused from its declared length, then while reading one with none.
    for _ in 0..2 {
        let error = transport.send(post(&base, "{}")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Body);
        assert!(!error.is_retryable());
        let source = std::error::Error::source(&error).unwrap().to_string();
        assert_eq!(source, "response body exceeds 10 bytes");
    }
    server.await.unwrap();
}

#[tokio::test]
async fn connection_failures_are_retryable() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let transport = ReqwestTransport::new().unwrap();

    let error = transport
        .send(post(&format!("http://{address}/"), "{}"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Connect);
    assert!(error.is_retryable());
}

#[tokio::test]
async fn requests_without_an_absolute_url_are_rejected() {
    let transport = ReqwestTransport::new().unwrap();

    let error = transport
        .send(post("/v1/responses", "{}"))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Request);
    assert!(!error.is_retryable());
}

#[test]
fn debug_output_shows_the_body_limit_only() {
    let transport = ReqwestTransport::from(reqwest::Client::new()).with_max_body_bytes(1_024);
    assert_eq!(
        format!("{transport:?}"),
        "ReqwestTransport { max_body_bytes: 1024, .. }"
    );
}

/// Run with `cargo test -p conveyor-reqwest -- --ignored` to check TLS and
/// certificate verification against a real endpoint. Sends no credentials.
#[tokio::test]
#[ignore = "needs network access"]
async fn tls_reaches_a_public_https_endpoint() {
    let transport = ReqwestTransport::new().unwrap();
    let request = Request::builder()
        .uri("https://api.openai.com/v1/models")
        .body(Bytes::new())
        .unwrap();

    let reply = timeout(Duration::from_secs(30), transport.send(request))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(reply.status(), StatusCode::UNAUTHORIZED);
}
