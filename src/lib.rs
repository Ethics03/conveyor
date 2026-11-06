//! Conveyor: a self-hosted runtime for persistent agents.
//!
//! The public entry point for Conveyor's workspace crates.
//!
//! Depend on a component crate directly when only that component is needed.
//! Domain and persistence errors remain separate so core consumers do not need
//! SQLite or a Tokio runtime.
//!
//! ```no_run
//! use conveyor::{Store, models::Session};
//!
//! # async fn example() -> conveyor::storage::Result<()> {
//! let store = Store::open(":memory:").await?;
//! let session = Session::default();
//! store.save_session(&session).await?;
//! store.close().await?;
//! # Ok(())
//! # }
//! ```

pub use conveyor_agent as agent;
pub use conveyor_core as core;
pub use conveyor_core::{Error as DomainError, models};
pub use conveyor_harness as harness;
pub use conveyor_http as http;
pub use conveyor_reqwest as reqwest;
pub use conveyor_storage as storage;
pub use conveyor_storage::{Error as StorageError, Store};
pub use conveyor_tools as tools;
