//! Steady-state `send_parts` and `recv_into` over `tcp_mesh` allocate nothing, and the
//! buffer `recv_into` hands back cycles through a fixed pair of allocations.
//!
//! The only test in this binary, so the counting allocator sees no other test.

use linalg_core::Transport;
use linalg_transport::tcp_mesh;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const WAIT: Duration = Duration::from_secs(20);
const LEN: usize = 256 << 10;
const WARMUP: usize = 50;
const MEASURED: usize = 1000;

#[test]
fn recv_into_reuses_capacity_and_steady_state_allocates_nothing() {
    let listeners: Vec<TcpListener> =
        (0..2).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    drop(listeners);

    let a = addrs.clone();
    let echo = std::thread::spawn(move || {
        let ep = tcp_mesh(1, &a, WAIT).unwrap();
        let mut buf = Vec::with_capacity(LEN);
        for _ in 0..WARMUP + MEASURED {
            ep.recv_into(0, &mut buf, WAIT).unwrap();
            assert_eq!(buf.len(), LEN);
            ep.send_parts(0, &[&buf[..8]]).unwrap();
        }
        ep
    });
    let ep = tcp_mesh(0, &addrs, WAIT).unwrap();
    let payload = vec![0x5Au8; LEN];
    let mut ack = Vec::with_capacity(8);
    let mut pointers = BTreeSet::new();
    let mut measured_allocs = 0;
    for i in 0..WARMUP + MEASURED {
        if i == WARMUP {
            measured_allocs = ALLOCS.load(Ordering::SeqCst);
        }
        let index = (i as u64).to_le_bytes();
        ep.send_parts(1, &[&index, &payload[8..]]).unwrap();
        ep.recv_into(1, &mut ack, WAIT).unwrap();
        assert_eq!(ack, index, "round trip {i}");
        if i >= WARMUP {
            pointers.insert(ack.as_ptr() as usize);
            assert!(ack.capacity() <= LEN, "ack buffer grew to {}", ack.capacity());
        }
    }
    let allocs = ALLOCS.load(Ordering::SeqCst) - measured_allocs;
    let _echo = echo.join().unwrap();
    assert_eq!(allocs, 0, "{allocs} allocations across {MEASURED} steady-state round trips");
    assert!(pointers.len() <= 2, "recv_into handed back {} distinct buffers", pointers.len());
}
