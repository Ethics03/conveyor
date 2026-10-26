//! Provider-neutral domain records, identifiers, and serialized state enums.

use chrono::{DateTime, FixedOffset, SubsecRound, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Map;
use uuid::Uuid;

use crate::{Error, Result};

/// An open-ended JSON object, used for tool arguments and durable metadata.
pub type JsonObject = Map<String, serde_json::Value>;
/// An offset-aware timestamp, preserving the original stored timezone offset.
/// Storage accepts microsecond precision; constructors generate that precision.
pub type Timestamp = DateTime<FixedOffset>;

// Each wire spelling is declared once for both Serde and SQL bindings.
macro_rules! string_enum {
    ($(#[$meta:meta])* pub enum $name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(#[serde(rename = $wire)] $variant),+
        }

        impl $name {
            /// The stable spelling used in persisted data and JSON.
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $wire),+ }
            }
        }
    };
}

string_enum! {
    pub enum SessionStatus { Active => "active", Archived => "archived" }
}
string_enum! {
    pub enum SessionTitleSource { Default => "default", Auto => "auto", User => "user" }
}
string_enum! {
    pub enum RunStatus {
        Pending => "pending", Running => "running", Blocked => "blocked",
        Finished => "finished", Cancelled => "cancelled", Failed => "failed"
    }
}
string_enum! {
    pub enum MessageRole {
        User => "user", Assistant => "assistant", System => "system", Tool => "tool"
    }
}
string_enum! {
    pub enum EventType {
        SessionCreated => "session.created", MessageCreated => "message.created",
        ContextCompactionPlanned => "context.compaction_planned",
        ContextCompacted => "context.compacted", RunStarted => "run.started",
        RunBlocked => "run.blocked", RunCancelled => "run.cancelled",
        RunResumed => "run.resumed", RunFinished => "run.finished",
        ToolStarted => "tool.started", ToolCancelled => "tool.cancelled",
        ApprovalRequested => "approval.requested", ApprovalResolved => "approval.resolved",
        ToolFinished => "tool.finished", RunFailed => "run.failed"
    }
}
string_enum! {
    pub enum ApprovalStatus { Pending => "pending", Approved => "approved", Denied => "denied" }
}
string_enum! {
    pub enum ApprovalDecision { Approved => "approved", Denied => "denied" }
}
string_enum! {
    pub enum ToolPermission { Read => "read", Write => "write", Dangerous => "dangerous" }
}

impl From<ApprovalDecision> for ApprovalStatus {
    fn from(decision: ApprovalDecision) -> Self {
        match decision {
            ApprovalDecision::Approved => Self::Approved,
            ApprovalDecision::Denied => Self::Denied,
        }
    }
}

/// Agent configuration; model and tool selection are resolved by the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub instructions: String,
    pub model: Option<String>,
    pub tools: Vec<String>,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            id: new_id("agent"),
            name: "Assistant".into(),
            instructions: String::new(),
            model: None,
            tools: Vec::new(),
        }
    }
}

/// A durable conversation, independent of any particular agent turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub title_source: SessionTitleSource,
    pub status: SessionStatus,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl Default for Session {
    fn default() -> Self {
        let now = now();
        Self {
            id: new_id("ses"),
            title: "New session".into(),
            title_source: SessionTitleSource::Default,
            status: SessionStatus::Active,
            created_at: now,
            updated_at: now,
        }
    }
}

impl Session {
    /// Create a session with a validated, user-authored title.
    ///
    /// Returns an error for an empty title or more than 100 Unicode characters.
    pub fn with_title(title: &str) -> Result<Self> {
        Ok(Self {
            title: clean_title(title)?,
            title_source: SessionTitleSource::User,
            ..Self::default()
        })
    }
}

/// A persisted agent run. Required identities have no empty defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub session_id: String,
    pub agent_id: String,
    pub parent_run_id: Option<String>,
    pub status: RunStatus,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub error: Option<String>,
}

impl Run {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        let now = now();
        Self {
            id: new_id("run"),
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            parent_run_id: None,
            status: RunStatus::Pending,
            created_at: now,
            updated_at: now,
            error: None,
        }
    }
}

/// A transcript message, including correlated tool calls and opaque metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub session_id: String,
    pub role: MessageRole,
    pub content: String,
    pub run_id: Option<String>,
    pub name: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub created_at: Timestamp,
    pub metadata: JsonObject,
}

impl Message {
    pub fn new(
        session_id: impl Into<String>,
        role: MessageRole,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: new_id("msg"),
            session_id: session_id.into(),
            role,
            content: content.into(),
            run_id: None,
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            created_at: now(),
            metadata: Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunOutcome {
    pub run: Run,
    pub final_message: Option<Message>,
    pub iterations: usize,
}

/// An append-only lifecycle record; `kind` serializes as the `type` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: EventType,
    pub session_id: Option<String>,
    pub run_id: Option<String>,
    pub message_id: Option<String>,
    pub payload: JsonObject,
    pub created_at: Timestamp,
}

impl Event {
    pub fn new(kind: EventType) -> Self {
        Self {
            id: new_id("evt"),
            kind,
            session_id: None,
            run_id: None,
            message_id: None,
            payload: Map::new(),
            created_at: now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonObject,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: JsonObject) -> Self {
        Self {
            id: new_id("call"),
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub ok: bool,
    pub content: String,
    pub metadata: JsonObject,
}

/// A request to authorize one exact tool call. Persisted decisions are immutable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub session_id: String,
    pub run_id: String,
    pub tool_call: ToolCall,
    pub reason: String,
    pub id: String,
    pub status: ApprovalStatus,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
}

impl ApprovalRequest {
    pub fn new(
        session_id: impl Into<String>,
        run_id: impl Into<String>,
        tool_call: ToolCall,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            run_id: run_id.into(),
            tool_call,
            reason: String::new(),
            id: new_id("appr"),
            status: ApprovalStatus::Pending,
            created_at: now(),
            resolved_at: None,
        }
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::now_v7().simple())
}

/// Current UTC time at the durable format's microsecond precision.
pub fn now() -> Timestamp {
    Utc::now().trunc_subsecs(6).fixed_offset()
}

/// Normalize a session title and reject empty or over-100-character titles.
pub fn clean_title(title: &str) -> Result<String> {
    let mut normalized = String::new();
    let mut length = 0;
    let mut pending_space = false;
    for character in title.chars() {
        // Treat the four ASCII separators as Unicode whitespace too.
        if character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character) {
            pending_space = !normalized.is_empty();
            continue;
        }
        length += 1 + usize::from(pending_space);
        if length > 100 {
            return Err(Error::TitleTooLong);
        }
        if pending_space {
            normalized.push(' ');
            pending_space = false;
        }
        normalized.push(character);
    }
    if normalized.is_empty() {
        return Err(Error::EmptyTitle);
    }
    Ok(normalized)
}
