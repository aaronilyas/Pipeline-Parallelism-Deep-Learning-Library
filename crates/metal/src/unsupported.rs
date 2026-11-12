//! Non-macOS build: no Metal, no dependencies.

/// Mirrors the one `linalg_core::Error` variant this build can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    UnsupportedDevice,
}

/// Placeholder: it cannot be constructed.
pub struct MetalBackend {
    _private: (),
}

impl MetalBackend {
    pub fn new() -> Result<MetalBackend, Error> {
        Err(Error::UnsupportedDevice)
    }
}
