//! Tool registration, stable advertised catalogs, allowlists, and result wrapping.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

use conveyor_core::{
    Cancelled,
    models::{JsonObject, ToolCall, ToolPermission, ToolResult},
    providers::ToolDefinition,
    tools::{DynTool, ToolContext, ToolError},
};

/// A tool with this name is already registered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Tool already registered: {0}")]
pub struct DuplicateTool(pub String);

/// Names that match no registered tool, sorted and without duplicates.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Unknown tools: {}", .0.join(", "))]
pub struct UnknownTools(pub Vec<String>);

/// An immutable set of tools, keyed and ordered by name.
///
/// Each tool's definition is captured when it is registered, so the catalog
/// advertised to a model and the tools dispatched for its calls stay the same
/// for the registry's lifetime. Cloning is cheap: tools are shared.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    entries: BTreeMap<String, Entry>,
}

#[derive(Clone)]
struct Entry {
    definition: ToolDefinition,
    permission: ToolPermission,
    parallel_safe: bool,
    tool: Arc<dyn DynTool>,
}

impl ToolRegistry {
    /// An empty registry.
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Add `tool` under the name in its definition.
    pub fn register(&mut self, tool: impl DynTool + 'static) -> Result<(), DuplicateTool> {
        let definition = tool.definition();
        if self.entries.contains_key(&definition.name) {
            return Err(DuplicateTool(definition.name));
        }
        let entry = Entry {
            permission: tool.permission(),
            parallel_safe: tool.parallel_safe(),
            definition,
            tool: Arc::new(tool),
        };
        self.entries.insert(entry.definition.name.clone(), entry);
        Ok(())
    }

    /// Add `tool` and return the registry, for building one in an expression.
    pub fn with(mut self, tool: impl DynTool + 'static) -> Result<Self, DuplicateTool> {
        self.register(tool)?;
        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Registered names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The permission that decides whether calls to `name` need approval.
    pub fn permission(&self, name: &str) -> Option<ToolPermission> {
        self.entries.get(name).map(|entry| entry.permission)
    }

    /// Whether calls to `name` may run concurrently with other parallel-safe calls.
    pub fn parallel_safe(&self, name: &str) -> Option<bool> {
        self.entries.get(name).map(|entry| entry.parallel_safe)
    }

    /// Definitions to advertise to a model, sorted by name.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.entries
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// The registry restricted to `names`, such as an agent's allowlist.
    ///
    /// Fails, listing every missing name, when any name is not registered.
    pub fn subset<I>(&self, names: I) -> Result<Self, UnknownTools>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let requested: BTreeSet<String> = names
            .into_iter()
            .map(|name| name.as_ref().to_owned())
            .collect();
        let missing: Vec<String> = requested
            .iter()
            .filter(|name| !self.entries.contains_key(*name))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(UnknownTools(missing));
        }
        let entries = requested
            .into_iter()
            .filter_map(|name| {
                let entry = self.entries.get(&name)?.clone();
                Some((name, entry))
            })
            .collect();
        Ok(Self { entries })
    }

    /// Execute one call and wrap its outcome as a result correlated to the call.
    ///
    /// Unknown tools, invalid arguments, and tool failures become failed
    /// results that the model can read. Cancellation is the only error: the
    /// caller decides how a cancelled turn ends.
    pub async fn execute(
        &self,
        call: &ToolCall,
        context: &ToolContext,
    ) -> Result<ToolResult, Cancelled> {
        let Some(entry) = self.entries.get(&call.name) else {
            return Ok(failed(call, format!("Unknown tool: {}", call.name)));
        };
        match entry.tool.execute(call.arguments.clone(), context).await {
            Ok(output) => {
                let mut metadata = JsonObject::new();
                metadata.insert("permission".into(), entry.permission.as_str().into());
                metadata.insert("content_type".into(), output.content_type().into());
                Ok(ToolResult {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                    ok: true,
                    content: output.content().into_owned(),
                    metadata,
                })
            }
            Err(ToolError::Cancelled(cancelled)) => Err(cancelled),
            Err(error) => Ok(failed(call, error.to_string())),
        }
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_set().entries(self.names()).finish()
    }
}

fn failed(call: &ToolCall, content: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        ok: false,
        content,
        metadata: JsonObject::new(),
    }
}
