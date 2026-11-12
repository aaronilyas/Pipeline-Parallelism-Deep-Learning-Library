//! Byte transports between pipeline ranks. Payloads are opaque.

use linalg_core::{Error, Transport};
use std::time::Duration;

/// In-process endpoint.
pub struct LoopbackEndpoint {
    rank: usize,
    world: usize,
}

impl Transport for LoopbackEndpoint {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world(&self) -> usize {
        self.world
    }

    fn send(&self, _dest: usize, _bytes: &[u8]) -> Result<(), Error> {
        Ok(())
    }

    fn recv_timeout(&self, _src: usize, _timeout: Duration) -> Result<Vec<u8>, Error> {
        Ok(Vec::new())
    }
}

pub fn loopback(world: usize) -> Result<Vec<LoopbackEndpoint>, Error> {
    Ok((0..world).map(|rank| LoopbackEndpoint { rank, world }).collect())
}

pub fn localhost_mesh(world: usize) -> Result<Vec<impl Transport>, Error> {
    loopback(world)
}
