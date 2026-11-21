//! Shared process group management, bounded stream collection, cleanup, and reaping.

use std::{io, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader},
    process::{ChildStdout, Command},
};

/// Why collecting a process's output failed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ProcessError {
    #[error(transparent)]
    Spawn(io::Error),
    #[error("process timed out")]
    TimedOut,
    #[error("invalid process output: {0}")]
    Output(String),
    #[error("{0}")]
    Failed(String),
}

/// One page of records from a process, plus whether more followed it.
#[derive(Debug)]
pub(crate) struct Page<T> {
    pub(crate) items: Vec<T>,
    /// Records seen, counting one past the page when there was one.
    pub(crate) observed: usize,
    /// Whether a record past the page was seen and the process was stopped.
    pub(crate) stopped_early: bool,
    /// The exit code; `None` when the process was stopped or killed by a signal.
    pub(crate) exit_code: Option<i32>,
    /// The start of standard error, trimmed and decoded lossily.
    pub(crate) errors: String,
}

/// Reads one record at a time from a process's standard output.
pub(crate) trait RecordFormat {
    type Item;

    /// The next record, `None` at the end of the output, or an error
    /// describing malformed output.
    async fn next(
        &mut self,
        reader: &mut Delimited<ChildStdout>,
    ) -> Result<Option<Self::Item>, String>;
}

/// How [`collect_page`] runs a process and judges its outcome.
pub(crate) struct PageRequest {
    pub(crate) offset: usize,
    pub(crate) limit: usize,
    pub(crate) timeout: Duration,
    /// Most bytes of standard error kept.
    pub(crate) max_error_bytes: usize,
}

/// Run `command` and collect the records from `offset` to `offset + limit`,
/// looking one record further to learn whether the output continues.
///
/// The process runs in its own process group with no standard input. It is
/// stopped as soon as the look-ahead record arrives, when it exceeds the
/// timeout, or when its output is malformed, and is always reaped before this
/// returns. Dropping the future kills it. The exit status is reported, not
/// judged: the caller decides which outcomes are failures.
pub(crate) async fn collect_page<F: RecordFormat>(
    mut command: Command,
    request: PageRequest,
    mut format: F,
) -> Result<Page<F::Item>, ProcessError> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command.spawn().map_err(ProcessError::Spawn)?;
    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        return Err(ProcessError::Failed(
            "process output pipes were not created".into(),
        ));
    };
    let stop_after = request
        .offset
        .saturating_add(request.limit)
        .saturating_add(1);

    let work = async {
        let collect = async {
            let mut reader = Delimited::new(stdout);
            let mut items = Vec::new();
            let mut observed = 0;
            loop {
                match format.next(&mut reader).await? {
                    Some(item) => {
                        observed += 1;
                        if observed > request.offset && items.len() < request.limit {
                            items.push(item);
                        }
                        if observed >= stop_after {
                            return Ok((items, observed, true));
                        }
                    }
                    None => return Ok((items, observed, false)),
                }
            }
        };
        let (collected, errors) = tokio::join!(
            async {
                let collected: Result<_, String> = collect.await;
                // Stop the process at once rather than wait for it to finish
                // the output nobody will read.
                if !matches!(collected, Ok((_, _, false))) {
                    let _ = child.start_kill();
                }
                collected
            },
            drain(stderr, request.max_error_bytes),
        );
        let status = child.wait().await;
        (collected, errors, status)
    };

    let outcome = tokio::time::timeout(request.timeout, work).await;
    let (collected, errors, status) = match outcome {
        Ok(finished) => finished,
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(ProcessError::TimedOut);
        }
    };
    let (items, observed, stopped_early) = collected.map_err(ProcessError::Output)?;
    let status = status.map_err(|error| ProcessError::Failed(error.to_string()))?;
    Ok(Page {
        items,
        observed,
        stopped_early,
        exit_code: status.code(),
        errors: String::from_utf8_lossy(&errors).trim().to_owned(),
    })
}

/// Read `stream` to its end, keeping at most `max_bytes` of it.
async fn drain(stream: impl AsyncRead + Unpin, max_bytes: usize) -> Vec<u8> {
    let mut kept = Vec::new();
    let mut stream = stream;
    let mut buffer = [0; 4_096];
    while let Ok(read) = stream.read(&mut buffer).await {
        if read == 0 {
            break;
        }
        let room = max_bytes.saturating_sub(kept.len());
        kept.extend_from_slice(&buffer[..read.min(room)]);
    }
    kept
}

/// A buffered stream read as delimited records of bounded size.
pub(crate) struct Delimited<R> {
    reader: BufReader<R>,
}

impl<R: AsyncRead + Unpin> Delimited<R> {
    fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
        }
    }

    /// The bytes before the next `delimiter`, or the remainder at the end of
    /// the stream; `None` when nothing is left. A record longer than
    /// `max_bytes` is an `InvalidData` error.
    pub(crate) async fn read_until(
        &mut self,
        delimiter: u8,
        max_bytes: usize,
    ) -> io::Result<Option<Vec<u8>>> {
        let mut record = Vec::new();
        loop {
            let available = self.reader.fill_buf().await?;
            if available.is_empty() {
                return Ok((!record.is_empty()).then_some(record));
            }
            let (part, found) = match available.iter().position(|&byte| byte == delimiter) {
                Some(end) => (&available[..end], true),
                None => (available, false),
            };
            if record.len() + part.len() > max_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "process emitted an oversized record",
                ));
            }
            record.extend_from_slice(part);
            let consumed = part.len() + usize::from(found);
            self.reader.consume(consumed);
            if found {
                return Ok(Some(record));
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::time::{Duration, Instant};

    use tokio::process::{ChildStdout, Command};

    use super::{Delimited, PageRequest, ProcessError, RecordFormat, collect_page};

    /// Newline-terminated lines.
    struct Lines;

    impl RecordFormat for Lines {
        type Item = String;

        async fn next(
            &mut self,
            reader: &mut Delimited<ChildStdout>,
        ) -> Result<Option<String>, String> {
            let line = reader
                .read_until(b'\n', 16)
                .await
                .map_err(|error| error.to_string())?;
            Ok(line.map(|line| String::from_utf8_lossy(&line).into_owned()))
        }
    }

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    fn request(offset: usize, limit: usize, timeout: Duration) -> PageRequest {
        PageRequest {
            offset,
            limit,
            timeout,
            max_error_bytes: 8,
        }
    }

    #[tokio::test]
    async fn a_full_page_stops_the_process_without_waiting_for_it() {
        let started = Instant::now();
        let page = collect_page(
            shell("printf 'a\\nb\\nc\\nd\\n'; exec sleep 30"),
            request(1, 1, Duration::from_secs(20)),
            Lines,
        )
        .await
        .unwrap();

        assert_eq!(page.items, ["b"]);
        assert_eq!(page.observed, 3);
        assert!(page.stopped_early);
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn output_that_ends_reports_every_record_exit_code_and_errors() {
        let page = collect_page(
            shell("printf 'a\\nb'"),
            request(0, 5, Duration::from_secs(20)),
            Lines,
        )
        .await
        .unwrap();
        assert_eq!(page.items, ["a", "b"]);
        assert_eq!(page.observed, 2);
        assert!(!page.stopped_early);

        let failed = collect_page(
            shell("echo 'went wrong badly' >&2; exit 3"),
            request(0, 5, Duration::from_secs(20)),
            Lines,
        )
        .await
        .unwrap();
        assert_eq!(failed.exit_code, Some(3));
        assert_eq!(failed.errors, "went wro");
        assert_eq!(page.exit_code, Some(0));
    }

    #[tokio::test]
    async fn slow_processes_time_out_and_bad_output_is_refused() {
        let started = Instant::now();
        let error = collect_page(
            shell("exec sleep 30"),
            request(0, 5, Duration::from_millis(200)),
            Lines,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, ProcessError::TimedOut));
        assert!(started.elapsed() < Duration::from_secs(10));

        let error = collect_page(
            shell("printf '%040d\\n' 0; exec sleep 30"),
            request(0, 5, Duration::from_secs(20)),
            Lines,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&error, ProcessError::Output(message) if message == "process emitted an oversized record")
        );
    }
}
