# Conveyor

A self-hosted runtime for persistent agents, implemented in Rust.

The first runtime milestone is implemented: domain records and asynchronous
SQLite storage for sessions, runs, messages, events, and approvals. The agent
loop, providers, tools, and runtime lifecycle remain subsequent steps.

Storage uses an owned async worker that serializes accepted operations, finishes
their transactions if callers stop waiting, and drains them on explicit close.
The version 1 schema and serialized record formats are stable. Unknown schema
versions and timestamps beyond the format's microsecond precision are rejected.
Single-statement writes use implicit transactions; related state/event writes
commit together in explicit transactions. Filtered event and approval queries use
direct predicates so SQLite can use their existing session/run indexes.

## Structure

Conveyor is a Cargo workspace. The root `conveyor` crate re-exports the component
crates; consumers can also depend on a component directly.

```text
conveyor/
├── Cargo.toml                  # workspace and public conveyor package
├── Cargo.lock                  # shared resolved dependencies
├── src/lib.rs                  # public re-exports
├── crates/
│   ├── conveyor-core/          # domain records and provider contracts
│   ├── conveyor-http/          # async HTTP transport contracts
│   ├── conveyor-reqwest/       # reqwest transport implementation
│   ├── conveyor-storage/      # async SQLite persistence
│   ├── conveyor-tools/        # concrete tools and registry
│   ├── conveyor-agent/        # turns, context, approvals, and tool orchestration
│   └── conveyor-harness/      # session ownership, interruption, and shutdown
├── test-support/fixtures/     # shared versioned SQL/JSON fixtures
├── examples/
└── docs/plans/rust-runtime-plan.md
```

Domain records in `conveyor-core` and persistence in `conveyor-storage` are
implemented, with their tests inside each crate. The other crates and provider
modules are scaffolds. Add implementations and dependencies as their migration
steps arrive. `conveyor-http` defines transport contracts; `conveyor-reqwest`
will implement them. Provider wire conversion belongs in `conveyor-core`.

Domain and storage APIs expose their own typed `Error` and `Result`. The facade
re-exports the errors as `DomainError` and `StorageError`; it does not combine
them into a shared error enum that would couple core consumers to SQLite.

The dependency direction and implementation milestones are in the
[runtime plan](docs/plans/rust-runtime-plan.md).

## Development

Use fish. `rust-toolchain.toml` selects the tested Rust 1.99.1 compiler, rustfmt,
and Clippy; resolved dependencies are committed in `Cargo.lock`.

Runtime and I/O APIs are async. Pure record construction, validation, and encoding
remain synchronous. The library exposes typed errors using `thiserror`;
`anyhow` can provide reporting context at a future executable boundary.

```fish
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked --all-targets
cargo test --workspace --locked --doc
env RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --locked --no-deps
```

The test suite runs entirely in Rust without model credentials. It covers
transaction rollback, approval conflicts, full-queue cancellation and draining,
indexed query plans, shutdown, database reopen,
and reading/updating versioned SQL/JSON fixtures while preserving timestamps,
tool-call correlation, and opaque provider metadata.

Root Cargo commands select every member by default; CI also selects the entire
workspace explicitly. All members inherit the version, edition, compiler
requirement, and publish setting from the root manifest.

Development PRs target `dev`. The workspace lives at the repository root; `main`
remains the stable branch.

The next milestone is a complete fake-provider turn: create a session, persist
input, execute the provider loop, and persist the final response and events.
