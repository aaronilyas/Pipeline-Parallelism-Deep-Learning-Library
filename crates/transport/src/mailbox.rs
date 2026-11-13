//! A FIFO of owned messages with a blocking, timed pop.

use linalg_core::Error;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    queue: VecDeque<Vec<u8>>,
}

#[derive(Default)]
pub(crate) struct Mailbox {
    state: Mutex<State>,
    ready: Condvar,
}

impl Mailbox {
    fn lock(&self) -> Result<MutexGuard<'_, State>, Error> {
        self.state.lock().map_err(|_| Error::Transport)
    }

    pub(crate) fn push(&self, message: Vec<u8>) -> Result<(), Error> {
        self.lock()?.queue.push_back(message);
        self.ready.notify_one();
        Ok(())
    }

    /// Pops the oldest message, waiting up to `timeout`. An empty mailbox at
    /// the deadline is `Error::Transport`.
    pub(crate) fn pop_timeout(&self, timeout: Duration) -> Result<Vec<u8>, Error> {
        let deadline = Instant::now().checked_add(timeout);
        let mut state = self.lock()?;
        loop {
            if let Some(message) = state.queue.pop_front() {
                return Ok(message);
            }
            state = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(Error::Transport);
                    }
                    self.ready.wait_timeout(state, deadline - now).map_err(|_| Error::Transport)?.0
                }
                None => self.ready.wait(state).map_err(|_| Error::Transport)?,
            };
        }
    }
}
