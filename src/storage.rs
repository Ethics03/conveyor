//! SQLite storage ownership, domain encoding, and transactional persistence.
//!
//! All database operations are asynchronous. An owned worker serializes accepted
//! requests so that dropping a caller does not interrupt a transaction. Explicit
//! close drains accepted requests before closing the SQLite connection.
//!
//! ```no_run
//! use conveyor::{Store, models::{Message, MessageRole, Session}};
//!
//! # async fn example() -> conveyor::Result<()> {
//! let store = Store::open("conveyor.sqlite").await?;
//! let session = Session::with_title("Investigate a timeout")?;
//! store.save_session(&session).await?;
//! let message = Message::new(&session.id, MessageRole::User, "Inspect the logs");
//! store.save_message(&message).await?;
//! assert_eq!(store.list_messages(&session.id).await?, vec![message]);
//! store.close().await?;
//! # Ok(())
//! # }
//! ```

mod rows;
mod store;

pub use store::Store;
