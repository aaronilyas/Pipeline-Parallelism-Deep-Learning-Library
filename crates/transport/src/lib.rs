//! Byte transports between pipeline ranks. Payloads are opaque.

mod loopback;
mod mailbox;
mod tcp;

pub use loopback::{loopback, LoopbackEndpoint};
pub use tcp::{localhost_mesh, tcp_mesh, DEFAULT_QUEUE_DEPTH, HANDSHAKE_MAGIC};
