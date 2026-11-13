//! The two per-peer queues of a TCP endpoint, each with a free list so that buffers are
//! recycled instead of reallocated.
//!
//! [`Outbox`]: application `send` → writer thread, bounded, blocking when full.
//! [`Inbox`]: reader thread → application `recv`, unbounded read-ahead.

use linalg_core::Error;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one of these locks leaves the state consistent: every critical
    // section is a few field updates.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct OutState {
    queue: VecDeque<Vec<u8>>,
    free: Vec<Vec<u8>>,
    /// Queued messages plus slots reserved by senders that are filling a buffer.
    slots: usize,
    /// The connection is unusable; every later send fails.
    failed: bool,
    /// The endpoint is dropping: the writer drains the queue and exits.
    closed: bool,
}

pub(crate) struct Outbox {
    depth: usize,
    state: Mutex<OutState>,
    not_full: Condvar,
    not_empty: Condvar,
}

impl Outbox {
    pub(crate) fn new(depth: usize) -> Self {
        Outbox {
            depth,
            state: Mutex::new(OutState {
                queue: VecDeque::with_capacity(depth),
                free: Vec::with_capacity(depth + 1),
                slots: 0,
                failed: false,
                closed: false,
            }),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        }
    }

    /// Queues the concatenation of `parts`: one copy, into a recycled buffer. Blocks while
    /// `depth` messages are queued.
    pub(crate) fn push_parts(&self, parts: &[&[u8]]) -> Result<(), Error> {
        let total = parts
            .iter()
            .try_fold(0usize, |sum, part| sum.checked_add(part.len()))
            .ok_or(Error::Transport)?;
        let mut buf = {
            let mut state = lock(&self.state);
            while !state.failed && state.slots >= self.depth {
                state = self.not_full.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.failed {
                return Err(Error::Transport);
            }
            state.slots += 1;
            state.free.pop().unwrap_or_default()
        };
        buf.clear();
        buf.reserve(total);
        for part in parts {
            buf.extend_from_slice(part);
        }
        let mut state = lock(&self.state);
        if state.failed {
            state.slots -= 1;
            return Err(Error::Transport);
        }
        state.queue.push_back(buf);
        drop(state);
        self.not_empty.notify_one();
        Ok(())
    }

    /// The writer's next message, or `None` once the outbox has failed, or is closed and
    /// drained.
    pub(crate) fn pop(&self) -> Option<Vec<u8>> {
        let mut state = lock(&self.state);
        loop {
            if state.failed {
                return None;
            }
            if let Some(message) = state.queue.pop_front() {
                state.slots -= 1;
                drop(state);
                self.not_full.notify_one();
                return Some(message);
            }
            if state.closed {
                return None;
            }
            state = self.not_empty.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Returns a written buffer for reuse.
    pub(crate) fn recycle(&self, buf: Vec<u8>) {
        let mut state = lock(&self.state);
        if state.free.len() <= self.depth {
            state.free.push(buf);
        }
    }

    pub(crate) fn fail(&self) {
        let mut state = lock(&self.state);
        state.failed = true;
        state.queue.clear();
        drop(state);
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    pub(crate) fn close(&self) {
        lock(&self.state).closed = true;
        self.not_empty.notify_all();
    }
}

/// Buffers kept for the reader beyond those in flight.
const INBOX_FREE_MAX: usize = 8;

struct InState {
    ready: VecDeque<Vec<u8>>,
    free: Vec<Vec<u8>>,
    /// No further messages will arrive; queued ones can still be taken.
    closed: bool,
}

pub(crate) struct Inbox {
    state: Mutex<InState>,
    ready: Condvar,
}

impl Inbox {
    pub(crate) fn new() -> Self {
        Inbox {
            state: Mutex::new(InState {
                ready: VecDeque::new(),
                free: Vec::with_capacity(INBOX_FREE_MAX),
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    /// A buffer for the reader to fill, preferring a recycled one.
    pub(crate) fn take_free(&self) -> Vec<u8> {
        lock(&self.state).free.pop().unwrap_or_default()
    }

    pub(crate) fn push(&self, message: Vec<u8>) {
        lock(&self.state).ready.push_back(message);
        self.ready.notify_one();
    }

    pub(crate) fn close(&self) {
        lock(&self.state).closed = true;
        self.ready.notify_all();
    }

    /// Waits up to `timeout` for the oldest message and hands it to `take` under the lock.
    fn pop_with<R>(
        &self,
        timeout: Duration,
        take: impl FnOnce(&mut InState, Vec<u8>) -> R,
    ) -> Result<R, Error> {
        let deadline = Instant::now().checked_add(timeout);
        let mut state = lock(&self.state);
        loop {
            if let Some(message) = state.ready.pop_front() {
                return Ok(take(&mut state, message));
            }
            if state.closed {
                return Err(Error::Transport);
            }
            state = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(Error::Transport);
                    }
                    self.ready
                        .wait_timeout(state, deadline - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                }
                None => self.ready.wait(state).unwrap_or_else(|e| e.into_inner()),
            };
        }
    }

    pub(crate) fn pop_timeout(&self, timeout: Duration) -> Result<Vec<u8>, Error> {
        self.pop_with(timeout, |_, message| message)
    }

    /// Swaps the oldest message into `buf`; `buf`'s old allocation goes to the free list.
    pub(crate) fn pop_into(&self, buf: &mut Vec<u8>, timeout: Duration) -> Result<(), Error> {
        self.pop_with(timeout, |state, mut message| {
            std::mem::swap(buf, &mut message);
            if state.free.len() < INBOX_FREE_MAX {
                state.free.push(message);
            }
        })
    }
}
