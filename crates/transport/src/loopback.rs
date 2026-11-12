//! In-process transport: one mailbox per (destination, source) pair.

use crate::mailbox::Mailbox;
use linalg_core::{Error, Transport};
use std::sync::Arc;
use std::time::Duration;

struct Hub {
    world: usize,
    /// Indexed `dest * world + src`.
    boxes: Vec<Mailbox>,
}

/// One rank of an in-process mesh.
pub struct LoopbackEndpoint {
    rank: usize,
    hub: Arc<Hub>,
}

/// Creates `world` endpoints, ranks `0..world`. `world < 2` is `Error::Transport`.
pub fn loopback(world: usize) -> Result<Vec<LoopbackEndpoint>, Error> {
    if world < 2 {
        return Err(Error::Transport);
    }
    let hub =
        Arc::new(Hub { world, boxes: (0..world * world).map(|_| Mailbox::default()).collect() });
    Ok((0..world).map(|rank| LoopbackEndpoint { rank, hub: Arc::clone(&hub) }).collect())
}

impl Transport for LoopbackEndpoint {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world(&self) -> usize {
        self.hub.world
    }

    fn send(&self, dest: usize, bytes: &[u8]) -> Result<(), Error> {
        if dest >= self.hub.world || dest == self.rank {
            return Err(Error::Transport);
        }
        self.hub.boxes[dest * self.hub.world + self.rank].push(bytes.to_vec())
    }

    fn recv_timeout(&self, src: usize, timeout: Duration) -> Result<Vec<u8>, Error> {
        if src >= self.hub.world || src == self.rank {
            return Err(Error::Transport);
        }
        self.hub.boxes[self.rank * self.hub.world + src].pop_timeout(timeout)
    }
}
