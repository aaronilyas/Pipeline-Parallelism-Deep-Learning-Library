//! State that `&self` calls and `Store` drops share across threads: the size-keyed free list of
//! device buffers, and the pinned host staging that `download_into`, `upload_into`, and
//! `mse_grad` copy through.
//!
//! This is the only file of the crate that may hold a `Mutex` outside the counters' guard (the
//! workspace audit enforces it). gemm, gelu, and gelu_backward never take these locks; alloc,
//! upload, a `Store` drop, and the transfers do.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Device buffers no tensor owns any more, keyed by their byte size.
pub(crate) struct FreeList<B> {
    free: Mutex<HashMap<usize, Vec<B>>>,
}

impl<B> FreeList<B> {
    pub(crate) fn new() -> Self {
        Self { free: Mutex::new(HashMap::new()) }
    }

    /// A free buffer of exactly `bytes`, if one was returned earlier.
    pub(crate) fn take(&self, bytes: usize) -> Option<B> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner).get_mut(&bytes)?.pop()
    }

    /// Returns `buffer` of `bytes` to the list.
    pub(crate) fn put(&self, bytes: usize, buffer: B) {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(bytes)
            .or_default()
            .push(buffer);
    }
}

/// Exclusive use of the staging buffers.
pub(crate) struct Exclusive<S> {
    inner: Mutex<S>,
}

impl<S> Exclusive<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self { inner: Mutex::new(inner) }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, S> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
