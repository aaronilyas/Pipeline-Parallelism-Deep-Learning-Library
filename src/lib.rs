//! `linalg`: Metal-accelerated dense linear algebra primitives for Apple Silicon.
//!
//! [`abi`] holds the frozen data model shared by every primitive; [`ops`] holds the
//! primitives themselves. Every primitive works on zero-copy unified-memory buffers and
//! exposes an `encode_*` function (records GPU work into a caller-owned command buffer)
//! plus a synchronous wrapper that commits and waits.

pub mod abi;
pub mod ops;
