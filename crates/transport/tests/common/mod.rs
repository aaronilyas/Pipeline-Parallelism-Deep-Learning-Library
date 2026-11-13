//! Ports for tests that need addresses before anything listens on them.

use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicU16, Ordering};

/// Below the ephemeral range of Linux (32768), macOS and Windows (49152), so no outgoing
/// connection on this host takes one of these ports as its source port.
const FIRST: u16 = 20000;
const SPAN: u16 = 10000;

static NEXT: AtomicU16 = AtomicU16::new(0);

/// `n` distinct ports on 127.0.0.1 that were free when checked, none handed out twice by this
/// process. Each test process starts at a different offset.
pub fn free_addrs(n: usize) -> Vec<SocketAddr> {
    let offset = (std::process::id() % u32::from(SPAN)) as u16;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let step = NEXT.fetch_add(1, Ordering::Relaxed);
        let port = FIRST + (offset.wrapping_add(step.wrapping_mul(7)) % SPAN);
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            out.push(listener.local_addr().unwrap());
        }
    }
    out
}
