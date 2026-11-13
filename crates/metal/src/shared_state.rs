//! State that `&self` calls share across threads: the last committed command buffer, which
//! `sync`, `download`, `download_into`, `upload_into`, and `mse_grad` wait on, and the one
//! buffer `mse_grad` writes the loss into.
//!
//! This is the only file of the crate that may hold a `Mutex` outside the counters' guard (the
//! workspace audit enforces it). Each lock covers a pointer swap or a clone; no lock is held
//! while the GPU is waited on, except `mse_grad`'s, which serialises use of its loss buffer.

use std::sync::{Mutex, MutexGuard, PoisonError};

pub(crate) struct Shared<C, L> {
    last: Mutex<Option<C>>,
    loss: Mutex<L>,
}

impl<C, L> Shared<C, L> {
    pub(crate) fn new(loss: L) -> Self {
        Self { last: Mutex::new(None), loss: Mutex::new(loss) }
    }

    /// Makes `cmd` the last committed command buffer and returns the one it replaces, so the
    /// caller drops (releases) it after the lock is gone.
    pub(crate) fn set_last(&self, cmd: C) -> Option<C> {
        self.last.lock().unwrap_or_else(PoisonError::into_inner).replace(cmd)
    }

    /// Runs `f` on the last committed command buffer, if any, under the lock.
    pub(crate) fn with_last<R>(&self, f: impl FnOnce(Option<&C>) -> R) -> R {
        f(self.last.lock().unwrap_or_else(PoisonError::into_inner).as_ref())
    }

    /// Exclusive use of the loss buffer.
    pub(crate) fn loss(&self) -> MutexGuard<'_, L> {
        self.loss.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
