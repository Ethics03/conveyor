use std::{
    convert::Infallible,
    sync::atomic::{AtomicUsize, Ordering},
};

use conveyor_core::{
    CancellationToken, Cancelled,
    models::{JsonObject, ToolCall, ToolPermission},
    tools::{Tool, ToolContext, ToolError},
};
use conveyor_tools::{DuplicateTool, ToolRegistry, UnknownTools};
use serde::Deserialize;
use serde_json::{Value, json};

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

fn context() -> ToolContext {
    ToolContext::new(".", CancellationToken::new())
}

#[derive(Deserialize)]
struct LookupArgs {
    key: String,
}

#[derive(Debug, thiserror::Error)]
enum LookupError {
    #[error("No entry for {0}")]
    Missing(String),
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}

#[derive(Default)]
struct Lookup {
    calls: AtomicUsize,
}

impl Tool for Lookup {
    const NAME: &'static str = "lookup";
    type Args = LookupArgs;
    type Output = Value;
    type Error = LookupError;

    fn description(&self) -> String {
        "Look up a key".into()
    }

    fn parameters(&self) -> JsonObject {
        object(json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: LookupError) -> ToolError {
        match error {
            LookupError::Cancelled(cancelled) => cancelled.into(),
            other => ToolError::failed(other),
        }
    }

    async fn call(&self, args: LookupArgs, context: &ToolContext) -> Result<Value, LookupError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match args.key.as_str() {
            "wait" => Err(context.cancellation().cancelled().await.into()),
            "missing" => Err(LookupError::Missing(args.key)),
            key => Ok(json!({"key": key, "value": 1})),
        }
    }
}

#[derive(Deserialize)]
struct NoArgs {}

struct Erase;

impl Tool for Erase {
    const NAME: &'static str = "erase";
    type Args = NoArgs;
    type Output = &'static str;
    type Error = Infallible;

    fn description(&self) -> String {
        "Erase everything".into()
    }

    fn parameters(&self) -> JsonObject {
        object(json!({"type": "object", "properties": {}}))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Dangerous
    }

    async fn call(
        &self,
        _args: NoArgs,
        _context: &ToolContext,
    ) -> Result<&'static str, Infallible> {
        Ok("erased")
    }
}

fn registry() -> ToolRegistry {
    ToolRegistry::new()
        .with(Lookup::default())
        .unwrap()
        .with(Erase)
        .unwrap()
}

#[test]
fn tools_are_listed_and_advertised_in_name_order() {
    let registry = registry();

    assert_eq!(registry.len(), 2);
    assert_eq!(registry.names().collect::<Vec<_>>(), ["erase", "lookup"]);
    let definitions = registry.definitions();
    assert_eq!(definitions[0].name, "erase");
    assert_eq!(definitions[1].name, "lookup");
    assert_eq!(definitions[1].description, "Look up a key");
    assert_eq!(definitions[1].parameters["required"], json!(["key"]));
    assert_eq!(
        registry.permission("erase"),
        Some(ToolPermission::Dangerous)
    );
    assert_eq!(registry.parallel_safe("lookup"), Some(true));
    assert_eq!(registry.parallel_safe("erase"), Some(false));
    assert_eq!(registry.permission("other"), None);
    assert_eq!(format!("{registry:?}"), r#"{"erase", "lookup"}"#);
}

#[test]
fn a_name_can_be_registered_once() {
    let mut registry = registry();

    assert_eq!(registry.register(Erase), Err(DuplicateTool("erase".into())));
    assert_eq!(registry.len(), 2);
}

#[test]
fn subsets_keep_the_named_tools_and_report_every_unknown_name() {
    let registry = registry();

    let subset = registry.subset(["lookup", "lookup"]).unwrap();
    assert_eq!(subset.names().collect::<Vec<_>>(), ["lookup"]);
    assert!(registry.subset(Vec::<String>::new()).unwrap().is_empty());
    assert_eq!(
        registry
            .subset(["write", "lookup", "bash", "write"])
            .unwrap_err(),
        UnknownTools(vec!["bash".into(), "write".into()])
    );
}

#[tokio::test]
async fn successful_calls_become_correlated_results() {
    let registry = registry();
    let call = ToolCall::new("lookup", object(json!({"key": "a"})));

    let result = registry.execute(&call, &context()).await.unwrap();

    assert_eq!(result.tool_call_id, call.id);
    assert_eq!(result.name, "lookup");
    assert!(result.ok);
    assert_eq!(result.content, r#"{"key":"a","value":1}"#);
    assert_eq!(
        Value::Object(result.metadata),
        json!({"permission": "read", "content_type": "application/json"})
    );

    let erase = ToolCall::new("erase", JsonObject::new());
    let result = registry.execute(&erase, &context()).await.unwrap();
    assert!(result.ok);
    assert_eq!(result.content, "erased");
    assert_eq!(
        Value::Object(result.metadata),
        json!({"permission": "dangerous", "content_type": "text/plain"})
    );
}

#[tokio::test]
async fn unknown_tools_bad_arguments_and_failures_become_failed_results() {
    let registry = registry();
    let cases = [
        (
            ToolCall::new("missing_tool", JsonObject::new()),
            "Unknown tool: missing_tool",
        ),
        (
            ToolCall::new("lookup", object(json!({"key": 7}))),
            "Invalid arguments for lookup: invalid type: integer `7`, expected a string",
        ),
        (
            ToolCall::new("lookup", object(json!({"key": "missing"}))),
            "No entry for missing",
        ),
    ];

    for (call, content) in cases {
        let result = registry.execute(&call, &context()).await.unwrap();
        assert_eq!(result.tool_call_id, call.id);
        assert_eq!(result.name, call.name);
        assert!(!result.ok);
        assert_eq!(result.content, content);
        assert!(result.metadata.is_empty());
    }
}

#[tokio::test]
async fn cancellation_is_returned_rather_than_reported_as_a_result() {
    let lookup = Lookup::default();
    let registry = ToolRegistry::new();
    let registry = registry.with(lookup).unwrap();

    // Already cancelled: the tool never runs.
    let cancellation = CancellationToken::new();
    cancellation.cancel("stop");
    let call = ToolCall::new("lookup", object(json!({"key": "a"})));
    let cancelled = registry
        .execute(&call, &ToolContext::new(".", cancellation))
        .await
        .unwrap_err();
    assert_eq!(cancelled.reason(), "stop");

    // Cancelled while running.
    let cancellation = CancellationToken::new();
    let context = ToolContext::new(".", cancellation.clone());
    let call = ToolCall::new("lookup", object(json!({"key": "wait"})));
    let (outcome, ()) = tokio::join!(registry.execute(&call, &context), async {
        tokio::task::yield_now().await;
        cancellation.cancel("interrupted");
    });
    assert_eq!(outcome.unwrap_err().reason(), "interrupted");
}
