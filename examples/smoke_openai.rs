//! A live agent session against OpenAI using Conveyor's read-only tools.
//!
//! Needs `OPENAI_API_KEY`, and reads `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, and
//! `OPENAI_PROJECT_ID` when set. Uses `gpt-5.3-codex` unless `OPENAI_MODEL`
//! names another model, and the model's default reasoning effort unless
//! `OPENAI_REASONING_EFFORT` sets one, such as `medium`. From the repository
//! root:
//!
//! ```text
//! cargo run --example smoke_openai
//! ```
//!
//! The agent answers two questions about this repository in one session,
//! reading files with tools for the first and building on that context for
//! the second. Each run makes a few small, billed API calls.

use std::{env, error::Error, sync::Arc};

use conveyor::{
    Store,
    core::providers::{
        ReplayState,
        openai::{OpenAiConfig, OpenAiProvider, ReasoningEffort},
    },
    harness::Runtime,
    models::{Agent, MessageRole},
    reqwest::ReqwestTransport,
    tools::default_registry,
};
use serde_json::Value;

const SMOKE_MODEL: &str = "gpt-5.3-codex";

const QUESTIONS: [&str; 2] = [
    "Which crates does this Cargo workspace contain? Check the manifests.",
    "Which of those crates is the only one allowed to run SQL? Use what you already read, \
     and check a file only if you must.",
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut config = OpenAiConfig::from_env().ok_or("Set OPENAI_API_KEY to run this example")?;
    config.model = env::var("OPENAI_MODEL").unwrap_or_else(|_| SMOKE_MODEL.into());
    if let Ok(effort) = env::var("OPENAI_REASONING_EFFORT") {
        let effort: ReasoningEffort = serde_json::from_value(Value::String(effort)).map_err(
            |_| "OPENAI_REASONING_EFFORT must be none, minimal, low, medium, high, xhigh, or max",
        )?;
        config.reasoning_effort = Some(effort);
    }
    println!(
        "model: {}, reasoning effort: {}",
        config.model,
        config
            .reasoning_effort
            .map_or("model default", ReasoningEffort::as_str)
    );

    let provider = Arc::new(OpenAiProvider::new(
        config,
        Arc::new(ReqwestTransport::new()?),
    )?);
    let workspace = env::current_dir()?;
    let runtime = Runtime::with_tools(
        Store::open(":memory:").await?,
        provider,
        default_registry(),
        workspace,
    );
    let session = runtime.create_session(Some("OpenAI smoke run")).await?;
    let agent = Agent {
        name: "Explorer".into(),
        instructions: "You explore a Rust repository with read-only tools. Check facts with \
                       the tools instead of guessing, and answer in one or two sentences."
            .into(),
        tools: vec![
            "read_file".into(),
            "read_many".into(),
            "search_files".into(),
        ],
        ..Agent::default()
    };

    for question in QUESTIONS {
        println!("\n> {question}");
        let outcome = runtime
            .run_turn(agent.clone(), &session.id, question)
            .await?;
        let messages = runtime.store().list_messages(&session.id).await?;
        for message in messages
            .iter()
            .filter(|message| message.run_id.as_deref() == Some(outcome.run.id.as_str()))
        {
            match message.role {
                MessageRole::Assistant => {
                    let saved = message
                        .metadata
                        .get(ReplayState::METADATA_KEY)
                        .and_then(ReplayState::from_metadata)
                        .map(|state| {
                            state
                                .items
                                .iter()
                                .filter_map(|item| item.get("type").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    println!("  model output [{saved}]");
                    for call in &message.tool_calls {
                        let arguments = Value::Object(call.arguments.clone());
                        println!("  tool call    {} {arguments}", call.name);
                    }
                }
                MessageRole::Tool => {
                    let ok = message.metadata.get("ok") == Some(&Value::Bool(true));
                    println!(
                        "  tool result  {} ({} chars)",
                        if ok { "ok" } else { "failed" },
                        message.content.chars().count()
                    );
                }
                _ => {}
            }
        }
        println!(
            "  run {:?} after {} model call(s)",
            outcome.run.status, outcome.iterations
        );
        if let Some(error) = &outcome.run.error {
            println!("  error: {error}");
        }
        if let Some(answer) = &outcome.final_message {
            println!("  answer: {}", answer.content);
            if let Some(usage) = answer.metadata.get("usage") {
                println!("  last call usage: {usage}");
            }
        }
    }
    runtime.close().await?;
    Ok(())
}
