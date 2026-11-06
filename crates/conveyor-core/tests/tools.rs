use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use conveyor_core::{
    CancellationToken, Cancelled,
    models::{JsonObject, ToolPermission},
    tools::{DynTool, Tool, ToolContext, ToolError, ToolOutput},
};
use serde::Deserialize;
use serde_json::{Value, json};

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("expected an object, got {other}"),
    }
}

#[derive(Deserialize)]
struct CountArgs {
    path: String,
    #[serde(default)]
    fail: bool,
}

#[derive(Debug, thiserror::Error)]
enum CountError {
    #[error("No such file: {0}")]
    Missing(String),
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}

#[derive(Default)]
struct Count {
    calls: AtomicUsize,
}

impl Tool for Count {
    const NAME: &'static str = "count";
    type Args = CountArgs;
    type Output = Value;
    type Error = CountError;

    fn description(&self) -> String {
        "Count lines".into()
    }

    fn parameters(&self) -> JsonObject {
        object(json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "fail": {"type": "boolean"}},
            "required": ["path"],
        }))
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Read
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn map_error(&self, error: CountError) -> ToolError {
        match error {
            CountError::Cancelled(cancelled) => cancelled.into(),
            other => ToolError::failed(other),
        }
    }

    async fn call(&self, args: CountArgs, context: &ToolContext) -> Result<Value, CountError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if args.fail {
            return Err(CountError::Missing(args.path));
        }
        if args.path == "slow" {
            return Err(context.cancellation().cancelled().await.into());
        }
        Ok(json!({"path": args.path, "lines": 3}))
    }
}

struct Write;

impl Tool for Write {
    const NAME: &'static str = "write";
    type Args = JsonObject;
    type Output = &'static str;
    type Error = Infallible;

    fn description(&self) -> String {
        "Write".into()
    }

    fn parameters(&self) -> JsonObject {
        JsonObject::new()
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::Write
    }

    async fn call(
        &self,
        _args: JsonObject,
        _context: &ToolContext,
    ) -> Result<&'static str, Infallible> {
        Ok("written")
    }
}

fn context() -> ToolContext {
    ToolContext::new("/workspace", CancellationToken::new())
}

#[test]
fn erased_tools_expose_their_definition_permission_and_parallel_safety() {
    let tools: Vec<Arc<dyn DynTool>> = vec![Arc::new(Count::default()), Arc::new(Write)];
    let definition = tools[0].definition();
    assert_eq!(definition.name, "count");
    assert_eq!(definition.description, "Count lines");
    assert_eq!(definition.parameters["required"], json!(["path"]));
    assert_eq!(tools[0].permission(), ToolPermission::Read);
    assert!(tools[0].parallel_safe());
    assert_eq!(tools[1].permission(), ToolPermission::Write);
    assert!(!tools[1].parallel_safe());
}

#[tokio::test]
async fn execution_decodes_arguments_and_reports_output_content() {
    let context = context();
    assert_eq!(context.workspace(), std::path::Path::new("/workspace"));
    let output = DynTool::execute(
        &Count::default(),
        object(json!({"path": "a.txt"})),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(output.content_type(), "application/json");
    assert_eq!(output.content(), r#"{"lines":3,"path":"a.txt"}"#);

    let output = DynTool::execute(&Write, JsonObject::new(), &context)
        .await
        .unwrap();
    assert_eq!(output, ToolOutput::Text("written".into()));
    assert_eq!(output.content_type(), "text/plain");
}

#[tokio::test]
async fn invalid_arguments_and_tool_failures_are_model_visible_errors() {
    let tool = Count::default();
    let context = context();
    let error = DynTool::execute(&tool, object(json!({"path": 7})), &context)
        .await
        .unwrap_err();
    assert!(matches!(&error, ToolError::InvalidArguments { tool, .. } if tool == "count"));
    assert!(
        error
            .to_string()
            .starts_with("Invalid arguments for count: ")
    );
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);

    let error = DynTool::execute(&tool, object(json!({"path": "x", "fail": true})), &context)
        .await
        .unwrap_err();
    assert!(matches!(error, ToolError::Failed(_)));
    assert_eq!(error.to_string(), "No such file: x");
}

#[tokio::test]
async fn cancelled_contexts_skip_execution_and_cancellation_stays_distinct() {
    let tool = Count::default();
    let context = context();
    context.cancellation().cancel("Stop");
    let error = DynTool::execute(&tool, object(json!({"path": "a"})), &context)
        .await
        .unwrap_err();
    assert!(matches!(&error, ToolError::Cancelled(cancelled) if cancelled.reason() == "Stop"));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn tools_can_report_cancellation_observed_during_a_call() {
    let tool = Arc::new(Count::default());
    let context = context();
    let running = {
        let (tool, context) = (Arc::clone(&tool), context.clone());
        tokio::spawn(async move {
            DynTool::execute(tool.as_ref(), object(json!({"path": "slow"})), &context).await
        })
    };
    while tool.calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    context.cancellation().cancel("Interrupted");
    let error = running.await.unwrap().unwrap_err();
    assert!(
        matches!(&error, ToolError::Cancelled(cancelled) if cancelled.reason() == "Interrupted")
    );
}

#[test]
fn outputs_render_null_as_empty_and_encode_serializable_values() {
    assert_eq!(ToolOutput::Json(Value::Null).content(), "");
    assert_eq!(ToolOutput::from("plain").content(), "plain");
    let output = ToolOutput::json(&["a", "b"]).unwrap();
    assert_eq!(output.content(), r#"["a","b"]"#);
    assert_eq!(output.content_type(), "application/json");
}
