//! Chat in the terminal with a Conveyor agent backed by OpenAI.
//!
//! The agent can read and search files in the workspace but not change them
//! or run commands. The workspace is the first argument, or the current
//! directory. Needs `OPENAI_API_KEY`; `OPENAI_MODEL` picks the model
//! (`gpt-5.3-codex` by default) and `OPENAI_REASONING_EFFORT` the reasoning
//! effort.
//!
//! ```text
//! cargo run --example chat_openai -- path/to/repository
//! ```
//!
//! Ctrl-C stops a running turn; Ctrl-D or `/exit` quits. The conversation
//! is kept in memory for the length of the session.

use std::{
    env,
    error::Error,
    io::{Write, stdout},
    path::PathBuf,
    sync::Arc,
};

use conveyor::{
    Store,
    core::providers::openai::{OpenAiConfig, OpenAiProvider, ReasoningEffort},
    harness::Runtime,
    models::{Agent, MessageRole, RunOutcome, RunStatus},
    reqwest::ReqwestTransport,
    tools::default_registry,
};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader, stdin};

const CHAT_MODEL: &str = "gpt-5.3-codex";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut config = OpenAiConfig::from_env().ok_or("Set OPENAI_API_KEY to chat")?;
    config.model = env::var("OPENAI_MODEL").unwrap_or_else(|_| CHAT_MODEL.into());
    if let Ok(effort) = env::var("OPENAI_REASONING_EFFORT") {
        config.reasoning_effort = Some(serde_json::from_value(Value::String(effort)).map_err(
            |_| "OPENAI_REASONING_EFFORT must be none, minimal, low, medium, high, xhigh, or max",
        )?);
    }
    let workspace = match env::args().nth(1) {
        Some(path) => PathBuf::from(path).canonicalize()?,
        None => env::current_dir()?,
    };
    println!(
        "Conveyor chat · {} · reasoning {} · {}",
        config.model,
        config
            .reasoning_effort
            .map_or("default", ReasoningEffort::as_str),
        workspace.display()
    );
    println!("Ask about the code. Ctrl-C stops a running turn; Ctrl-D or /exit quits.");

    let provider = Arc::new(OpenAiProvider::new(
        config,
        Arc::new(ReqwestTransport::new()?),
    )?);
    let runtime = Runtime::with_tools(
        Store::open(":memory:").await?,
        provider,
        default_registry(),
        workspace,
    );
    let session = runtime.create_session(None).await?;
    let agent = Agent {
        name: "Assistant".into(),
        instructions: "You help with the code in your workspace. You can read and search \
                       files but not change them or run commands. Check facts with the tools \
                       instead of guessing, name the files you rely on, and keep answers short."
            .into(),
        tools: vec![
            "read_file".into(),
            "read_many".into(),
            "search_files".into(),
        ],
        ..Agent::default()
    };

    let mut lines = BufReader::new(stdin()).lines();
    loop {
        print!("\nyou> ");
        stdout().flush()?;
        let line = tokio::select! {
            line = lines.next_line() => line?,
            _ = tokio::signal::ctrl_c() => None,
        };
        let Some(line) = line else { break };
        let message = line.trim();
        if message.is_empty() {
            continue;
        }
        if message == "/exit" {
            break;
        }

        let outcome = runtime
            .start_turn(agent.clone(), &session.id, message)
            .await?
            .outcome();
        tokio::pin!(outcome);
        let outcome = tokio::select! {
            outcome = &mut outcome => outcome?,
            _ = tokio::signal::ctrl_c() => {
                runtime.interrupt(&session.id, "Stopped from the keyboard")?;
                outcome.await?
            }
        };
        report(&runtime, &outcome).await?;
    }
    println!();
    runtime.close().await?;
    Ok(())
}

/// Print the tools a turn used, its answer or why it ended, and its tokens.
async fn report(runtime: &Runtime, outcome: &RunOutcome) -> Result<(), Box<dyn Error>> {
    let messages = runtime
        .store()
        .list_messages(&outcome.run.session_id)
        .await?;
    let (mut input, mut cached, mut output) = (0, 0, 0);
    for message in messages
        .iter()
        .filter(|message| message.run_id.as_deref() == Some(outcome.run.id.as_str()))
    {
        match message.role {
            MessageRole::Assistant => {
                let usage = message.metadata.get("usage");
                let tokens = |pointer| {
                    usage
                        .and_then(|usage| usage.pointer(pointer))
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                };
                input += tokens("/input_tokens");
                cached += tokens("/input_tokens_details/cached_tokens");
                output += tokens("/output_tokens");
                for call in &message.tool_calls {
                    let arguments = Value::Object(call.arguments.clone());
                    println!("  · {} {arguments}", call.name);
                }
            }
            MessageRole::Tool => {
                let ok = message.metadata.get("ok") == Some(&Value::Bool(true));
                let size = message.content.chars().count();
                if ok {
                    println!("    ok, {size} chars");
                } else {
                    let reason = message.content.lines().next().unwrap_or_default();
                    println!("    failed: {reason}");
                }
            }
            _ => {}
        }
    }
    match outcome.run.status {
        RunStatus::Finished => {
            let answer = outcome
                .final_message
                .as_ref()
                .map_or("", |message| message.content.as_str());
            println!("\nagent> {answer}");
        }
        status => {
            let reason = outcome.run.error.as_deref().unwrap_or("no reason given");
            println!("\n(turn {}: {reason})", status.as_str());
        }
    }
    println!(
        "  [{} model call(s), {input} input tokens ({cached} cached), {output} output]",
        outcome.iterations
    );
    Ok(())
}
