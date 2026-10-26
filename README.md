# Conveyor

A self-hosted runtime for persistent agents, implemented in Rust.

The first runtime milestone is implemented: domain records and asynchronous
SQLite storage for sessions, runs, messages, events, and approvals. The agent
loop, providers, tools, and runtime lifecycle remain subsequent steps.

Storage uses an owned async worker that serializes accepted operations, finishes
their transactions if callers stop waiting, and drains them on explicit close.
The version 1 schema and serialized record formats are stable. Unknown schema
versions and timestamps beyond the format's microsecond precision are rejected.

## Structure

- `src/lib.rs`: crate root, public models, typed errors, and `Store`.
- `src/models.rs`, `error.rs`, `cancellation.rs`: domain types and runtime primitives.
- `src/runtime.rs`: session and turn ownership, interruption, and shutdown.
- `src/agent/`: provider loop, tool batches, and automatic titles.
- `src/approvals.rs`, `context.rs`: approval policy and derived prompt context.
- `src/providers/`: fake and Anthropic adapters.
- `src/tools/`: registry, workspace tools, shell execution, and process helpers.
- `src/storage/`: SQLite operations, row conversion, and `schema.sql`.
- `tests/`: model/storage integration suites and versioned storage fixtures.
- `examples/`: runnable examples as complete runtime paths become available.
- `docs/plans/rust-runtime-plan.md`: architecture, Rust conventions, and milestones.

The module boundaries and implementation milestones are in the
[runtime plan](docs/plans/rust-runtime-plan.md).

## Development

Use fish. `rust-toolchain.toml` selects the tested Rust 1.99.1 compiler, rustfmt,
and Clippy; resolved dependencies are committed in `Cargo.lock`.

Runtime and I/O APIs are async. Pure record construction, validation, and encoding
remain synchronous. The library exposes typed errors using `thiserror`;
`anyhow` can provide reporting context at a future executable boundary.

```fish
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
env RUSTDOCFLAGS=-Dwarnings cargo doc --locked --no-deps
```

The test suite runs entirely in Rust without model credentials. It covers
transaction rollback, approval conflicts, cancellation, shutdown, database reopen,
and reading/updating versioned SQL/JSON fixtures while preserving timestamps,
tool-call correlation, and opaque provider metadata.

Development PRs target `dev`. Rust source lives at the repository root; `main`
remains the stable branch.

The next milestone is a complete fake-provider turn: create a session, persist
input, execute the provider loop, and persist the final response and events.
