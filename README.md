# Conveyor

A self-hosted, async-first runtime for multi-agent orchestration, written in Rust.

Conveyor is built around one pattern: an **orchestrator agent** that stays with
the user while **subagents** do long work in the background. The orchestrator
hands off a task and gets control back right away. It keeps talking to the user,
checks on running tasks, steers or cancels them, and picks up their results as
they finish. The work the system can take on grows with the number of subagents,
not with the length of one agent loop.

```text
            user
              │  stays reachable
              ▼
     ┌──────────────────┐   start · check · update · cancel · list
     │   orchestrator   │ ─────────────────────────────────────────┐
     └──────────────────┘                                          │
              ▲  results and questions, delivered between model calls
              │                                                    ▼
     ┌────────┴─────────┬──────────────────┬──────────────────┐
     │    subagent A    │    subagent B    │    subagent C    │  each in its own
     │   (researcher)   │     (coder)      │    (reviewer)    │  session and task
     └──────────────────┴──────────────────┴──────────────────┘
```

## Design

- **Nothing blocks the orchestrator.** Every agent turn runs as its own task owned
  by the runtime, so starting a subagent returns at once and sessions never wait
  on each other.
- **Results come to the orchestrator.** A finished subagent reports back whether
  the orchestrator is busy (the result arrives before its next model call), idle
  (it is woken with a new turn), or waiting on an approval (the result waits too).
- **Subagents can be steered.** An update either interrupts the subagent and
  restarts it with the new instruction, or queues the instruction until the
  current run finishes.
- **Everything is durable.** Sessions, runs, messages, approvals, and an
  append-only event log live in SQLite, so a restart recovers what was in flight.
- **Cancellation is cooperative and complete.** Interrupting a turn stops a
  provider call in flight, saves the cancelled outcome, and leaves the session
  ready for the next turn.
- **Built for throughput.** Many agents share one provider and one store. Tokio
  tasks are cheap, so the limits that matter are provider rate limits and
  database writes, and the design budgets for both.

## Status

Conveyor is early. The foundation for the orchestrator is in place; the subagent
tools are next after parity with the earlier Python implementation.

| Area | Status |
| --- | --- |
| Durable sessions, runs, messages, events, and approvals (SQLite) | Done |
| Provider contract, fake provider, Anthropic adapter | Done |
| Agent run loop: provider calls, run outcomes, cancellation | Done |
| Runtime: each turn as a background task, one turn per session, interrupt, shutdown | Done |
| Built-in tools, approvals, and parallel tool execution | In progress |
| Live HTTP transport, OpenAI adapter, context planning, session titles | Planned |
| Subagent tools, result delivery, steering, and limits on depth, count, and budget | Planned |
| Shared provider rate limiting and higher-throughput storage | Planned |

## Example

The runtime API today runs single-agent turns, each in the background:

```rust
use std::sync::Arc;

use conveyor::{Store, core::providers::fake::FakeProvider, harness::Runtime, models::Agent};

async fn example() -> Result<(), Box<dyn std::error::Error>> {
    let store = Store::open("conveyor.sqlite").await?;
    let runtime = Runtime::new(store, Arc::new(FakeProvider::new(["Done."])));
    let session = runtime.create_session(Some("Release prep")).await?;

    // The turn runs as its own task; this one is free to start others.
    let turn = runtime
        .start_turn(Agent::default(), &session.id, "Draft the release notes")
        .await?;
    let outcome = turn.outcome().await?;
    println!("{:?}: {:?}", outcome.run.status, outcome.final_message);

    runtime.close().await?;
    Ok(())
}
```

## Workspace

The root `conveyor` crate re-exports the component crates; depend on a component
directly when that is all you need.

| Crate | Role |
| --- | --- |
| `conveyor-harness` | The runtime: sessions, background turns, interruption, shutdown |
| `conveyor-agent` | The agent loop: provider calls, run state, tool batches, approvals |
| `conveyor-tools` | Built-in tools and their registry |
| `conveyor-storage` | Async SQLite persistence |
| `conveyor-core` | Domain records, cancellation, provider and tool contracts, provider adapters |
| `conveyor-http` | The HTTP transport contract providers send through |
| `conveyor-reqwest` | The reqwest implementation of that transport |

Crate boundaries, the storage and execution contracts, and the milestones are in
the [runtime plan](docs/plans/rust-runtime-plan.md).

## Development

Use fish. `rust-toolchain.toml` pins the tested Rust 1.99.1 toolchain, and
`Cargo.lock` is committed.

```fish
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings -D clippy::wildcard_imports
cargo test --workspace --locked --all-targets
cargo test --workspace --locked --doc
env RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --locked --no-deps
```

The test suite needs no model credentials. Development PRs target `dev`; `main`
is the stable branch.
