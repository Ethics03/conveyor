//! Session and resource ownership for Conveyor hosts.
//!
//! Scaffold: the runtime will compose agent execution, providers, storage, and
//! tools. It owns turn admission, interruption, and orderly shutdown. It remains
//! independent of any CLI, server protocol, or UI.

mod runtime;
