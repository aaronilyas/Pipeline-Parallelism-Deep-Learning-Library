//! Metal backend for the device-independent linalg contract.
//!
//! On macOS this wraps the legacy `linalg` Metal kernels. On every other OS the crate has no
//! dependencies and [`MetalBackend::new`] reports `UnsupportedDevice`.

#[cfg(target_os = "macos")]
mod imp;
#[cfg(target_os = "macos")]
mod shared_state;
#[cfg(target_os = "macos")]
pub use imp::{MetalBackend, MetalStore};

#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(not(target_os = "macos"))]
pub use unsupported::{Error, MetalBackend};

#[cfg(test)]
mod tests;

#[cfg(all(test, target_os = "macos"))]
mod training;
