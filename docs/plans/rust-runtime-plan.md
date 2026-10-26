# Conveyor runtime: structure and implementation plan

Status: milestone 1 (domain records and async storage) is implemented.
The agent runtime, provider adapters, and concrete tools remain subsequent work.

## Product and scope

Conveyor is a self-hosted runtime for persistent agents. The runtime owns sessions,
model interaction, tool execution, approvals, durable transcripts, and events.
Desktop apps, gateways, and other interfaces are clients of this runtime.

Keep Rust source at the repository root. Develop on feature branches with PRs
targeting `dev`; `main` remains stable. Use fish for development commands.

The initial runtime feature set includes:

- Sessions, runs, messages, events, tool calls/results, and automatic titles.
- Async turns, one active turn per session, interruption, and orderly shutdown.
- Transactional SQLite state/event persistence and immutable approval decisions.
- Agent-specific tool allowlists and read/write/dangerous permissions.
- Complete-batch approval preflight, correlated denial results, and live waits.
- Bounded parallel tools, serial barriers, and stable result ordering.
- `read_file`, `read_many`, `search_files`, `write_file`, and `bash`.
- Context budgets, old tool-result clearing, and compaction telemetry.
- Fake and Anthropic providers, including caching and opaque compaction replay.

Desktop clients, a production gateway, sandbox/remote environments, memory,
skills, schedules, and multi-agent execution are later milestones. Persisted
approval records provide an audit trail; crash-resumable continuations and
exactly-once external side effects require separate designs.

## Package and modules

Start with one Cargo package and library crate named `conveyor`. Introduce
additional crates when independent consumers, dependency isolation, or build
costs justify them. Expose a small public API and keep helpers private or
`pub(crate)`. Add an executable when the host/client protocol is defined.

The current module layout is:

```text
conveyor/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── README.md
├── .github/workflows/ci.yml
├── docs/plans/rust-runtime-plan.md
├── src/
│   ├── lib.rs
│   ├── models.rs
│   ├── error.rs
│   ├── cancellation.rs
│   ├── runtime.rs
│   ├── approvals.rs
│   ├── context.rs
│   ├── agent.rs
│   ├── agent/
│   │   ├── run.rs
│   │   ├── tool_batch.rs
│   │   └── titles.rs
│   ├── providers.rs
│   ├── providers/
│   │   ├── fake.rs
│   │   └── anthropic.rs
│   ├── tools.rs
│   ├── tools/
│   │   ├── registry.rs
│   │   ├── workspace.rs
│   │   ├── shell.rs
│   │   └── process.rs
│   ├── storage.rs
│   └── storage/
│       ├── store.rs
│       ├── store/tests.rs
│       ├── rows.rs
│       └── schema.sql
├── tests/
│   ├── models.rs
│   ├── storage.rs
│   ├── storage_format.rs
│   ├── support/
│   └── fixtures/storage-v1/
│       ├── records.json
│       └── database.sql
└── examples/
```

Domain and storage modules are implemented. The other source modules currently
document their intended boundaries. Add working behavior, integration suites,
and runnable examples incrementally. No placeholder passing tests or fake APIs.

| Module | Owns | Boundary |
| --- | --- | --- |
| `models` | Domain records and serialized state enums | No database, filesystem, HTTP, or runtime dependencies |
| `providers` | Provider contract, model limits, request/response/replay types | No store access or tool execution |
| `tools` | Tool metadata, typed arguments, registry, context, concrete execution | No approval prompts, provider calls, or SQL |
| `approvals` | Allow/ask/deny policy and async decision contract | Orchestration persists decisions and dispatches tools |
| `context` | Pure derivation of bounded provider requests | Preserves durable history; no HTTP or SQL |
| `storage` | Schema, encoding, transactions, queries, connection lifecycle | Only production module that executes SQL |
| `agent` | Run transitions, provider iterations, approval coordination, tool batches | Uses contracts without direct filesystem/HTTP/SQL |
| `runtime` | Shared resources, sessions, active turns, title coordination, shutdown | Client-facing library API without a UI/transport dependency |

Register each tool's schema, permission, parallel-safety flag, and executor as one
definition. Retain the advertised catalog throughout a run's dispatch so the
executed tool matches the definition sent to the model. An immutable registry
is sufficient initially. Keep typed arguments/results inside concrete tools and
erase them only at the dynamic registry boundary.

Use dynamic dispatch for interchangeable providers, tools, and approval resolvers,
with boxed `Send` futures where an object-safe async boundary needs them. Keep
`Store` concrete. Add traits at real substitution boundaries; avoid generic
repositories, an event bus, and an environment framework before they are needed.

## Rust conventions

Follow the [Rust Style Guide](https://doc.rust-lang.org/style-guide/) and
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

- Use edition 2024 and the tested toolchain pinned in `rust-toolchain.toml`.
  `Cargo.toml` declares the supported compiler version. Commit `Cargo.lock`.
- Use `snake_case` for files, modules, functions, and methods; `UpperCamelCase`
  for types and traits; and `SCREAMING_SNAKE_CASE` for constants/statics.
- Use named module roots such as `providers.rs` with children in `providers/`.
  Both that form and `mod.rs` are valid; use the selected layout consistently.
- Keep visibility narrow and public re-exports deliberate. Plain data records
  can expose fields; resource owners keep their internals private.
- Use enums for statuses, permissions, finish reasons, and decisions. Preserve
  stable wire spellings with Serde. Use typed structs for known records and tool
  arguments; reserve `serde_json::Value` for open-ended provider/metadata payloads.
- Borrow inputs when ownership is unnecessary. Own data when storing it or
  moving it into a task. Introduce clones and `Arc` for actual ownership needs.
- Use conventional constructors and conversion traits. Add `Default` only for
  valid meaningful states, and options/builders when configuration needs them.
- Return `Result` for recoverable failures and `Option` for legitimate absence.
  Preserve sources with `thiserror` in library errors. `anyhow` can add reporting
  context at a future executable boundary. Avoid panics for external input.
- Keep runtime and I/O entrypoints async, including storage, provider calls,
  tools, approvals, interruption coordination, and shutdown. Pure constructors,
  validation, encoding, and transition decisions remain synchronous.
- Never hide `block_on` or a nested executor behind a synchronous I/O method.
  Keep blocking work off Tokio workers and blocking mutex guards out of awaits.
- Track spawned work. Review cancellation safety at every task, `select!`, queue,
  transaction, and shutdown boundary. A synchronous `Drop` cannot await cleanup.
- Document public APIs with rustdoc, including meaningful error/cancellation
  behavior. Use compiled examples and `#[must_use]` where dropping a result would
  silently discard intended work.
- Use default stable rustfmt and Clippy's standard warnings as errors. Explain
  narrow lint exceptions; avoid blanket suppression or every pedantic restriction.
- Prefer safe Rust. Isolate any necessary unsafe platform code and document its
  safety invariants. Review each working slice for both correctness and idiom.

## Async storage contract

One owned Tokio worker serializes operations on a SQLx SQLite connection through
a bounded queue. SQLx performs blocking SQLite calls on its own background
thread. File databases use WAL, foreign keys, and a five-second busy timeout.
Builds do not require a live database or compile-time query macros.

An accepted operation commits or rolls back even if its caller stops waiting.
Cancellation before enqueueing submits no work. Explicit `Store::close` stops
admission, drains work, and awaits the worker. Cancelling a close wait retains
the worker handle so another close attempt can finish the drain.

Runtime cancellation must be cooperative: await durable writes before reporting
a terminal outcome. Subsequent queued cleanup sees completed transactions. A
single-connection pool alone would not establish this operation-level contract.
See [SQLx connection behavior](https://docs.rs/sqlx/latest/sqlx/struct.SqliteConnection.html)
and [pool cancellation behavior](https://docs.rs/sqlx/latest/sqlx/struct.Pool.html).

State changes and associated events commit atomically. Keep events append-only.
Preserve version 1 table names, serialized enum/event values, JSON object shapes,
prefixed UUIDv7 identifiers, timezone offsets, and microsecond timestamp encoding.
Reject finer timestamp precision and incompatible/unversioned databases before
changing their schema or journal mode. Queries order tied timestamps by ID.

Versioned SQL/JSON fixtures verify stored formats, tool-call/result correlation,
opaque provider metadata, lifecycle updates, and reopen behavior. Fixture tests
use disposable databases and require no external interpreter or credentials.

## Execution and approval contract

`Runtime` owns shared resources and the active-turn registry. Allow one active
turn per session, with separate cancellation signals. Use a guard to release
bookkeeping on every exit path; await async cleanup and terminal persistence
explicitly. Keep state transitions in `agent/run.rs` testable independently of I/O.

Preflight the complete tool batch. Read-only tools may run automatically; write
and dangerous tools require an async approval decision. Missing resolvers fail
closed. Resolve every required approval before any tool in that batch begins.
Return correlated results for denials rather than executing side effects.

Run contiguous parallel-safe tools under a bound. Flush batches at serial and
denied-call barriers, then persist results in the model's original call order.
Distinguish tool errors from fatal/cancellation outcomes. Test completion racing
cancellation so finished work is not mislabeled as cancelled.

Shutdown rejects new turns, signals cancellation, awaits task/process cleanup,
persists terminal state, and closes resources. Shell/search processes require
explicit process-group termination, bounded stream draining, and reaping. The
initial implementation targets Unix; other platforms need explicit support/tests.

## Design references

Codex is the primary behavioral reference for tool execution and lifecycle
correctness. Its [tool orchestrator](https://github.com/openai/codex/blob/8c076942cb70fba8e5c4732c23f1faabc4aed484/codex-rs/core/src/tools/orchestrator.rs)
separates policy/approval from execution, and its
[parallel execution layer](https://github.com/openai/codex/blob/8c076942cb70fba8e5c4732c23f1faabc4aed484/codex-rs/core/src/tools/parallel.rs)
retains advertised tool context and tests cancellation races. Borrow these
boundaries while keeping Conveyor proportional to its current scope.

Rig is a secondary reference for provider/tool APIs. Its
[harness](https://github.com/0xPlaygrounds/rig/blob/5bada84e36ab5b4530d3d2cd425cf148233851ae/crates/rig-harness/README.md)
uses a Bevy/ECS kernel and plugins. Its
[tools](https://github.com/0xPlaygrounds/rig/blob/5bada84e36ab5b4530d3d2cd425cf148233851ae/crates/rig-tools/src/lib.rs)
declare resource footprints and return structured facts. Adopting those crates
requires evaluating their contracts; a full ECS/plugin host is a separate choice.

Goose's [state machine](https://github.com/aaif-goose/goose/blob/34a222e3f2b95d8b4a43482b53c2cc957ecce26d/crates/goose-agent/src/machine.rs)
and [confirmation coordinator](https://github.com/aaif-goose/goose/blob/34a222e3f2b95d8b4a43482b53c2cc957ecce26d/crates/goose/src/agents/tool_confirmation_coordinator.rs)
illustrate explicit transitions, active-turn guards, and cancellable batch waits.
Use focused mechanisms rather than introducing a general operation pipeline.

These links pin reviewed revisions. They establish design references, not
dependency compatibility or performance claims.

## Implementation milestones

1. **Complete: domain and durable state.** Typed records/errors, schema checks,
   async storage, atomic state/events, immutable approvals, versioned format
   fixtures, cancellation-safe completion, and explicit close.
2. **Complete fake-provider turn.** Provider contract, fake provider, basic
   context construction, runtime ownership, and a plain-response loop. Persist
   input, response, run outcomes, and events; verify reopening the session.
3. **Tools and approvals.** Registry, five built-in tools, policy, async resolver,
   and initially sequential batches. Verify denied calls have no side effects.
4. **Cancellation and parallel execution.** Bounded batches, barriers, ordered
   results, interruption at every wait point, process cleanup, and shutdown.
5. **Context and live provider.** Budgets, result clearing, compaction telemetry
   and opaque replay, Anthropic transport/caching, and automatic session titles.
   Define retries/timeouts explicitly and verify requests against mock HTTP.
6. **Runtime readiness.** Full scenario coverage, documentation, runnable smoke
   examples, and measured overhead. Report platform/live-provider limits plainly.

Every step must compile and exercise real working behavior. Keep API credentials
out of the deterministic suite; report any live-provider checks separately.
Benchmark claims require comparable workloads and release build settings.

## Verification

Run from fish at the repository root:

```fish
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
env RUSTDOCFLAGS=-Dwarnings cargo doc --locked --no-deps
```

CI runs these checks for PRs targeting `dev` or `main`, and pushes to `dev`.
Meaningful tests cover transactional failure, conflicting decisions, metadata
preservation, timestamp/ID ordering, submitted-operation cancellation, interrupted
close, and versioned database reopen. Add execution/context/provider scenarios
with their corresponding milestones.
