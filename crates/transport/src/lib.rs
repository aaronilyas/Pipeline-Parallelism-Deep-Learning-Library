//! Byte transports between pipeline ranks. Payloads are opaque.
//!
//! [`loopback`] connects ranks of one process through memory. [`localhost_mesh`] connects ranks
//! of one process over 127.0.0.1. [`tcp_mesh`] builds one rank's endpoint per process, from the
//! address every rank listens on.

mod loopback;
mod mailbox;
mod mesh;
mod queue;
mod sockbuf;
mod tcp;

pub use loopback::{loopback, LoopbackEndpoint};
pub use mesh::{tcp_mesh, tcp_mesh_with_depth, HANDSHAKE_MAGIC};
pub use tcp::{localhost_mesh, DEFAULT_QUEUE_DEPTH};
