//! Typed tool contracts and their object-safe form.
//!
//! Tools implement [`Tool`]; callers dispatch through [`DynTool`], which decodes
//! JSON arguments and erases output and error types.

use std::{
    borrow::Cow,
    error::Error as StdError,
    future::Future,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    BoxFuture, CancellationToken, Cancelled,
    models::{JsonObject, ToolPermission},
    providers::{BoxError, ToolDefinition},
};

/// What a tool may use while executing one call.
#[derive(Debug, Clone)]
pub struct ToolContext {
    workspace: PathBuf,
    cancellation: CancellationToken,
}

impl ToolContext {
    /// Run tools inside `workspace`, observing the turn's `cancellation`.
    pub fn new(workspace: impl Into<PathBuf>, cancellation: CancellationToken) -> Self {
        Self {
            workspace: workspace.into(),
            cancellation,
        }
    }

    /// The directory that confines workspace paths.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// The turn's cancellation signal; long-running tools check or await it.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
}

/// A successful tool's model-visible output.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutput {
    /// Plain text (`text/plain`).
    Text(String),
    /// Structured output, rendered as compact JSON (`application/json`).
    Json(Value),
}

impl ToolOutput {
    /// Encode a serializable value as JSON output.
    pub fn json<T: Serialize + ?Sized>(value: &T) -> Result<Self, serde_json::Error> {
        serde_json::to_value(value).map(Self::Json)
    }

    /// The content sent to the model. JSON `null` is empty.
    pub fn content(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Json(Value::Null) => Cow::Borrowed(""),
            Self::Json(value) => Cow::Owned(value.to_string()),
        }
    }

    /// The output's MIME type.
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Text(_) => "text/plain",
            Self::Json(_) => "application/json",
        }
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

/// Why a tool call produced no output.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// The arguments did not match the tool's argument type.
    #[error("Invalid arguments for {tool}: {source}")]
    InvalidArguments {
        tool: String,
        source: serde_json::Error,
    },
    /// The tool ran and failed; the message is shown to the model.
    #[error("{0}")]
    Failed(BoxError),
    /// The turn was cancelled before or during execution.
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}

impl ToolError {
    /// Wrap a tool's own failure.
    pub fn failed(error: impl Into<BoxError>) -> Self {
        Self::Failed(error.into())
    }
}

/// A typed tool. Arguments arrive as JSON and decode into [`Tool::Args`].
pub trait Tool: Send + Sync + 'static {
    /// The unique name advertised to the model and used for dispatch.
    const NAME: &'static str;
    /// Typed arguments, decoded from the model's JSON object.
    type Args: DeserializeOwned + Send;
    /// Typed output, converted to model-visible content after the call.
    type Output: Into<ToolOutput>;
    /// The tool's own error, converted by [`map_error`](Self::map_error).
    type Error: StdError + Send + Sync + 'static;

    /// Model-facing description of what the tool does.
    fn description(&self) -> String;

    /// JSON Schema for [`Tool::Args`].
    fn parameters(&self) -> JsonObject;

    /// The permission that decides whether a call needs approval.
    fn permission(&self) -> ToolPermission;

    /// Whether calls may run concurrently with other parallel-safe calls.
    fn parallel_safe(&self) -> bool {
        false
    }

    /// Convert a typed error. Override to report cancellation as
    /// [`ToolError::Cancelled`] rather than as a failed result.
    fn map_error(&self, error: Self::Error) -> ToolError {
        ToolError::failed(error)
    }

    /// Execute one call. Long-running work should observe the context's
    /// cancellation token; dropping the future must not leave work running.
    fn call(
        &self,
        args: Self::Args,
        context: &ToolContext,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// An object-safe [`Tool`]. Implemented for every `Tool`; implement it directly
/// only for tools defined at runtime.
pub trait DynTool: Send + Sync {
    /// The definition advertised to the model.
    fn definition(&self) -> ToolDefinition;

    /// The permission that decides whether a call needs approval.
    fn permission(&self) -> ToolPermission;

    /// Whether calls may run concurrently with other parallel-safe calls.
    fn parallel_safe(&self) -> bool;

    /// Decode `arguments` and execute one call.
    ///
    /// Returns [`ToolError::Cancelled`] without running when the context is
    /// already cancelled.
    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        context: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>>;
}

impl<T: Tool> DynTool for T {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(T::NAME, self.description(), self.parameters())
    }

    fn permission(&self) -> ToolPermission {
        Tool::permission(self)
    }

    fn parallel_safe(&self) -> bool {
        Tool::parallel_safe(self)
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        context: &'a ToolContext,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            context.cancellation().check()?;
            let args = serde_json::from_value(Value::Object(arguments)).map_err(|source| {
                ToolError::InvalidArguments {
                    tool: T::NAME.to_owned(),
                    source,
                }
            })?;
            match self.call(args, context).await {
                Ok(output) => Ok(output.into()),
                Err(error) => Err(self.map_error(error)),
            }
        })
    }
}
