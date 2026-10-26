//! Agent orchestration over provider, tool, approval, and storage contracts.
//!
//! Scaffold: provider iterations, context planning, approvals, tool batches, and
//! titles will be implemented incrementally. This crate orchestrates contracts;
//! concrete HTTP, SQL, filesystem, and process operations belong to other crates.

mod approvals;
mod context;
mod run;
mod titles;
mod tool_batch;
