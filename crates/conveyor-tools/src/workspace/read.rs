//! `read_file` and `read_many`: paged, bounded reads of UTF-8 workspace files.

use std::{
    fs::{self, File},
    io::{self, Read},
    path::Path,
};

use conveyor_core::{
    models::{JsonObject, ToolPermission},
    tools::{Tool, ToolContext, ToolError},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{WorkspaceError, blocking, clamp, resolve};

const DEFAULT_READ_OFFSET: i64 = 1;
const DEFAULT_READ_LIMIT: i64 = 500;
const MAX_READ_LIMIT: usize = 2_000;
const MAX_READ_FILE_CHARS: usize = 100_000;
const MAX_READ_LINE_CHARS: usize = 2_000;
const MAX_READ_MANY_FILES: usize = 20;
const DEFAULT_READ_MANY_TOTAL_CHARS: i64 = 100_000;
const MAX_READ_MANY_TOTAL_CHARS: usize = 250_000;
const LINE_TRUNCATION_MARKER: &str = " ... [truncated]";
const READ_CHUNK_BYTES: usize = 8_192;

fn default_offset() -> i64 {
    DEFAULT_READ_OFFSET
}

fn default_limit() -> i64 {
    DEFAULT_READ_LIMIT
}

fn default_total_chars() -> i64 {
    DEFAULT_READ_MANY_TOTAL_CHARS
}

/// Arguments for [`ReadFile`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadFileArgs {
    pub path: String,
    /// The first line to return, from 1.
    #[serde(default = "default_offset")]
    pub offset: i64,
    /// Most lines to return, up to 2000.
    #[serde(default = "default_limit")]
    pub limit: i64,
}

/// Reads a UTF-8 text file in pages of numbered lines.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Args = ReadFileArgs;
    type Output = Value;
    type Error = WorkspaceError;

    fn description(&self) -> String {
        "Read a UTF-8 text file from the workspace with line pagination.".into()
    }

    fn parameters(&self) -> JsonObject {
        schema(json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: WorkspaceError) -> ToolError {
        map_error(error)
    }

    async fn call(
        &self,
        args: ReadFileArgs,
        context: &ToolContext,
    ) -> Result<Value, WorkspaceError> {
        let page = blocking(context, move |workspace| {
            read_text_file(
                workspace,
                &args.path,
                args.offset,
                args.limit,
                MAX_READ_FILE_CHARS,
            )
        })
        .await?;
        super::json(&page)
    }
}

/// Arguments for [`ReadMany`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadManyArgs {
    /// Files to read; only the first 20 are read.
    pub paths: Vec<String>,
    #[serde(default = "default_offset")]
    pub offset: i64,
    #[serde(default = "default_limit")]
    pub limit: i64,
    /// The character budget shared by every file, up to 250,000.
    #[serde(default = "default_total_chars")]
    pub max_total_chars: i64,
}

/// Reads several UTF-8 files under one character budget. A file that cannot
/// be read is reported beside the others rather than failing the call.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadMany;

impl Tool for ReadMany {
    const NAME: &'static str = "read_many";
    type Args = ReadManyArgs;
    type Output = Value;
    type Error = WorkspaceError;

    fn description(&self) -> String {
        "Read multiple UTF-8 workspace files in one bounded batch.".into()
    }

    fn parameters(&self) -> JsonObject {
        schema(json!({
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
                "max_total_chars": {"type": "integer"},
            },
            "required": ["paths"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: WorkspaceError) -> ToolError {
        map_error(error)
    }

    async fn call(
        &self,
        args: ReadManyArgs,
        context: &ToolContext,
    ) -> Result<Value, WorkspaceError> {
        let batch = blocking(context, move |workspace| Ok(read_many(workspace, &args))).await?;
        super::json(&batch)
    }
}

pub(super) fn map_error(error: WorkspaceError) -> ToolError {
    match error {
        WorkspaceError::Cancelled(cancelled) => cancelled.into(),
        other => ToolError::failed(other),
    }
}

pub(super) fn schema(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        _ => JsonObject::new(),
    }
}

/// One page of a file, as `read_file` returns it.
#[derive(Debug, Serialize)]
struct FilePage {
    path: String,
    content: String,
    offset: usize,
    limit: usize,
    total_lines: usize,
    total_lines_is_exact: bool,
    truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_offset: Option<usize>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    line_truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    hint: Option<String>,
}

#[derive(Debug, Serialize)]
struct ManyFiles {
    files: Vec<FilePage>,
    errors: Vec<FileError>,
    requested_count: usize,
    processed_count: usize,
    total_chars: usize,
    max_total_chars: usize,
    truncated: bool,
}

#[derive(Debug, Serialize)]
struct FileError {
    path: String,
    error: String,
}

fn read_many(workspace: &Path, args: &ReadManyArgs) -> ManyFiles {
    let selected = &args.paths[..args.paths.len().min(MAX_READ_MANY_FILES)];
    let char_limit = clamp(args.max_total_chars, 1, MAX_READ_MANY_TOTAL_CHARS);
    let mut files = Vec::new();
    let mut errors = Vec::new();
    let mut total_chars = 0;
    let mut budget_truncated = false;

    for path in selected {
        let remaining = char_limit - total_chars;
        if remaining == 0 {
            break;
        }
        match read_text_file(workspace, path, args.offset, args.limit, remaining) {
            Ok(page) => {
                budget_truncated |= page.truncated;
                total_chars += page.content.chars().count();
                files.push(page);
            }
            Err(error) => errors.push(FileError {
                path: path.clone(),
                error: error.to_string(),
            }),
        }
        if total_chars >= char_limit {
            break;
        }
    }

    let processed_count = files.len() + errors.len();
    ManyFiles {
        truncated: budget_truncated
            || args.paths.len() > selected.len()
            || processed_count < selected.len(),
        files,
        errors,
        requested_count: args.paths.len(),
        processed_count,
        total_chars,
        max_total_chars: char_limit,
    }
}

/// Read one page of numbered lines from `path`, within `max_chars` characters.
///
/// Lines longer than [`MAX_READ_LINE_CHARS`] are clipped with a marker. One
/// line past the page is looked at, so `total_lines` is exact only when the
/// end of the file was reached.
fn read_text_file(
    workspace: &Path,
    path: &str,
    offset: i64,
    limit: i64,
    max_chars: usize,
) -> Result<FilePage, WorkspaceError> {
    let resolved = resolve(workspace, path)?;
    let is_file = fs::metadata(&resolved.path).is_ok_and(|metadata| metadata.is_file());
    if !is_file {
        return Err(WorkspaceError::NotAFile(path.to_owned()));
    }
    let file = File::open(&resolved.path).map_err(|source| WorkspaceError::io(path, source))?;
    let mut lines = Lines::new(file);
    let failed = |error: io::Error| match error.kind() {
        io::ErrorKind::InvalidData => WorkspaceError::NotText(path.to_owned()),
        _ => WorkspaceError::io(path, error),
    };

    let page_offset = clamp(offset, 1, usize::MAX);
    let page_limit = clamp(limit, 1, MAX_READ_LIMIT);
    let char_limit = max_chars.clamp(1, MAX_READ_FILE_CHARS);
    let page_end = page_offset.saturating_add(page_limit - 1);

    let mut content = String::new();
    let mut content_chars = 0;
    let mut content_lines = 0;
    let mut total_lines = 0;
    let mut total_lines_is_exact = false;
    let mut char_truncated = false;
    let mut line_truncated = false;
    let mut partial_line_truncated = false;
    let mut next_offset = None;

    let mut line_number = 0;
    loop {
        let Some((line, clipped)) = lines.next_line().map_err(failed)? else {
            total_lines_is_exact = true;
            break;
        };
        line_number += 1;
        total_lines = line_number;
        if line_number < page_offset {
            continue;
        }

        let numbered = format!("{line_number}|{line}");
        let numbered_chars = numbered.chars().count();
        let separator = usize::from(content_lines > 0);
        if content_chars + separator + numbered_chars > char_limit {
            char_truncated = true;
            if content_lines == 0 {
                // Even the first line does not fit: return as much of it as does.
                partial_line_truncated = true;
                content = numbered.chars().take(char_limit).collect();
                if lines.has_more().map_err(failed)? {
                    total_lines += 1;
                    next_offset = Some(line_number + 1);
                } else {
                    total_lines_is_exact = true;
                }
            } else {
                next_offset = Some(line_number);
            }
            break;
        }

        if separator == 1 {
            content.push('\n');
        }
        content.push_str(&numbered);
        content_chars += separator + numbered_chars;
        content_lines += 1;
        line_truncated |= clipped;

        if line_number >= page_end {
            if lines.has_more().map_err(failed)? {
                total_lines += 1;
                next_offset = Some(line_number + 1);
            } else {
                total_lines_is_exact = true;
            }
            break;
        }
    }

    if total_lines_is_exact && total_lines > 0 && page_offset > total_lines {
        return Err(WorkspaceError::OffsetPastEnd {
            offset: page_offset,
            lines: total_lines,
        });
    }

    let mut hints = Vec::new();
    if let Some(next) = next_offset {
        hints.push(format!("Continue with offset={next}."));
    }
    if line_truncated {
        hints.push("Lines longer than 2,000 characters were clipped.".to_owned());
    }
    if partial_line_truncated {
        hints.push(
            "The first selected line exceeded the output budget; its omitted remainder cannot \
             be retrieved with a line offset."
                .to_owned(),
        );
    }

    Ok(FilePage {
        path: resolved.display(),
        content,
        offset: page_offset,
        limit: page_limit,
        total_lines,
        total_lines_is_exact,
        truncated: char_truncated || line_truncated || next_offset.is_some(),
        next_offset,
        line_truncated,
        hint: (!hints.is_empty()).then(|| hints.join(" ")),
    })
}

/// The lines of a UTF-8 stream, each kept to [`MAX_READ_LINE_CHARS`]
/// characters, decoded in bounded chunks so a long line is never held whole.
///
/// Lines end at `\n`; a `\r` before it is dropped. Invalid UTF-8 anywhere in
/// the stream, including in a clipped part of a line, is an `InvalidData`
/// error.
struct Lines<R> {
    decoder: Utf8Chunks<R>,
    chunk: String,
    position: usize,
}

impl<R: Read> Lines<R> {
    fn new(reader: R) -> Self {
        Self {
            decoder: Utf8Chunks::new(reader),
            chunk: String::new(),
            position: 0,
        }
    }

    /// Whether any text is left.
    fn has_more(&mut self) -> io::Result<bool> {
        self.fill()
    }

    /// Load the next chunk if the current one is used up; false at the end.
    fn fill(&mut self) -> io::Result<bool> {
        while self.position >= self.chunk.len() {
            match self.decoder.next_chunk()? {
                Some(chunk) => {
                    self.chunk = chunk;
                    self.position = 0;
                }
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// The next line and whether it was clipped.
    fn next_line(&mut self) -> io::Result<Option<(String, bool)>> {
        if !self.fill()? {
            return Ok(None);
        }
        let mut line = String::new();
        let mut kept = 0;
        let mut clipped = false;
        loop {
            if !self.fill()? {
                break;
            }
            let rest = &self.chunk[self.position..];
            let (segment, ended) = match rest.find('\n') {
                Some(end) => (&rest[..end], true),
                None => (rest, false),
            };
            self.position += segment.len() + usize::from(ended);
            let segment = if ended {
                segment.strip_suffix('\r').unwrap_or(segment)
            } else {
                segment
            };
            for character in segment.chars() {
                if kept == MAX_READ_LINE_CHARS {
                    clipped = true;
                    break;
                }
                line.push(character);
                kept += 1;
            }
            if ended {
                // A `\r` split from its `\n` across chunks.
                if !clipped && line.ends_with('\r') {
                    line.pop();
                }
                break;
            }
        }
        if clipped {
            line.push_str(LINE_TRUNCATION_MARKER);
        }
        Ok(Some((line, clipped)))
    }
}

/// Decodes a byte stream as UTF-8 in chunks, carrying a character split
/// between reads into the next chunk.
struct Utf8Chunks<R> {
    reader: R,
    carry: Vec<u8>,
}

impl<R: Read> Utf8Chunks<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            carry: Vec::new(),
        }
    }

    fn next_chunk(&mut self) -> io::Result<Option<String>> {
        loop {
            let mut bytes = std::mem::take(&mut self.carry);
            let start = bytes.len();
            bytes.resize(start + READ_CHUNK_BYTES, 0);
            let read = self.reader.read(&mut bytes[start..])?;
            bytes.truncate(start + read);
            if read == 0 {
                return match bytes.is_empty() {
                    true => Ok(None),
                    false => Err(invalid_utf8()),
                };
            }
            match String::from_utf8(bytes) {
                Ok(text) => return Ok(Some(text)),
                Err(error) => {
                    let utf8 = error.utf8_error();
                    if utf8.error_len().is_some() {
                        return Err(invalid_utf8());
                    }
                    let valid = utf8.valid_up_to();
                    let mut bytes = error.into_bytes();
                    self.carry = bytes.split_off(valid);
                    if valid > 0 {
                        return String::from_utf8(bytes)
                            .map(Some)
                            .map_err(|_| invalid_utf8());
                    }
                }
            }
        }
    }
}

fn invalid_utf8() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::{LINE_TRUNCATION_MARKER, Lines, MAX_READ_LINE_CHARS};

    /// Reads at most `size` bytes per call, to split characters and lines.
    struct Trickle<'a> {
        bytes: &'a [u8],
        size: usize,
    }

    impl std::io::Read for Trickle<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let count = self.size.min(buffer.len()).min(self.bytes.len());
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            Ok(count)
        }
    }

    fn lines(bytes: &[u8], size: usize) -> std::io::Result<Vec<(String, bool)>> {
        let mut lines = Lines::new(Trickle { bytes, size });
        let mut all = Vec::new();
        while let Some(line) = lines.next_line()? {
            all.push(line);
        }
        Ok(all)
    }

    #[test]
    fn lines_survive_any_read_boundary() {
        let text = "café\r\n🦀 crab\nlast";
        for size in 1..=text.len() {
            let all = lines(text.as_bytes(), size).unwrap();
            assert_eq!(
                all,
                [
                    ("café".to_owned(), false),
                    ("🦀 crab".to_owned(), false),
                    ("last".to_owned(), false)
                ],
                "read size {size}"
            );
        }
    }

    #[test]
    fn long_lines_are_clipped_by_characters() {
        let long = "é".repeat(MAX_READ_LINE_CHARS + 5);
        let all = lines(format!("{long}\nnext\n").as_bytes(), 7).unwrap();
        let expected = format!(
            "{}{LINE_TRUNCATION_MARKER}",
            "é".repeat(MAX_READ_LINE_CHARS)
        );
        assert_eq!(all, [(expected, true), ("next".to_owned(), false)]);
    }

    #[test]
    fn invalid_utf8_is_an_error_even_in_a_clipped_part() {
        let mut bytes = vec![b'a'; MAX_READ_LINE_CHARS + 10];
        bytes.push(0xff);
        bytes.push(b'\n');
        for size in [3, 8_192] {
            let error = lines(&bytes, size).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
        assert!(
            lines(&[b'a', 0xe2, 0x82], 2).is_err(),
            "truncated character"
        );
    }

    #[test]
    fn empty_streams_have_no_lines() {
        assert!(lines(b"", 4).unwrap().is_empty());
        assert_eq!(lines(b"\n", 4).unwrap(), [(String::new(), false)]);
    }
}
