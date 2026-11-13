//! `tcp_mesh`: one endpoint per process, handshake, queued sends, read-ahead receives.
//!
//! Multi-process tests re-run this test binary as a child: `child_process_entry` reads its role
//! from the environment and does nothing when the variables are absent.

use linalg_core::{Error, Transport};
use linalg_transport::{tcp_mesh, DEFAULT_QUEUE_DEPTH, HANDSHAKE_MAGIC};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ROLE_ENV: &str = "LINALG_TRANSPORT_CHILD_ROLE";
const RANK_ENV: &str = "LINALG_TRANSPORT_CHILD_RANK";
const ADDRS_ENV: &str = "LINALG_TRANSPORT_CHILD_ADDRS";
const MIB4: usize = 4 << 20;
const CONNECT: Duration = Duration::from_secs(20);
const WAIT: Duration = Duration::from_secs(20);

/// Distinct free ports on 127.0.0.1, released before return.
fn free_addrs(n: usize) -> Vec<SocketAddr> {
    let listeners: Vec<TcpListener> =
        (0..n).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    listeners.iter().map(|l| l.local_addr().unwrap()).collect()
}

/// Bytes that differ at every offset, so a shifted or reordered message never matches.
fn pattern(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn hello(world: usize, rank: usize) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&HANDSHAKE_MAGIC);
    out[4..8].copy_from_slice(&(world as u32).to_le_bytes());
    out[8..12].copy_from_slice(&(rank as u32).to_le_bytes());
    out
}

fn dial(addr: SocketAddr) -> TcpStream {
    let deadline = Instant::now() + CONNECT;
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => return s,
            Err(e) if Instant::now() < deadline => {
                let _ = e;
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("dial {addr}: {e}"),
        }
    }
}

/// Accepts one connection, failing the test instead of blocking past `CONNECT`.
fn accept(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + CONNECT;
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false).unwrap();
                return s;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "nobody dialed");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}

/// Reads one frame (little-endian `u64` length, then payload) from a raw socket.
fn read_frame(s: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 8];
    s.read_exact(&mut len).unwrap();
    let mut out = vec![0u8; u64::from_le_bytes(len) as usize];
    s.read_exact(&mut out).unwrap();
    out
}

/// Kills the child when dropped, so a failing test leaves no process behind.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl ChildGuard {
    fn wait_until(&mut self, deadline: Instant) -> ExitStatus {
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "child did not exit in time");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn spawn_child(role: &str, rank: usize, addrs: &[SocketAddr]) -> ChildGuard {
    let list: Vec<String> = addrs.iter().map(ToString::to_string).collect();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["child_process_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role)
        .env(RANK_ENV, rank.to_string())
        .env(ADDRS_ENV, list.join(","))
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    ChildGuard(child)
}

/// Every rank sends one small and one 4 MiB message to every other rank and checks what it gets.
fn exchange(rank: usize, addrs: &[SocketAddr]) {
    let ep = tcp_mesh(rank, addrs, CONNECT).unwrap();
    let world = addrs.len();
    assert_eq!((ep.rank(), ep.world()), (rank, world));
    let big = pattern(MIB4 + world * world);
    for peer in (0..world).filter(|&p| p != rank) {
        ep.send(peer, format!("{rank}->{peer}").as_bytes()).unwrap();
        let off = rank * world + peer;
        ep.send_parts(peer, &[&[rank as u8], &big[off..off + MIB4]]).unwrap();
    }
    let mut buf = Vec::new();
    for peer in (0..world).filter(|&p| p != rank) {
        assert_eq!(ep.recv_timeout(peer, WAIT).unwrap(), format!("{peer}->{rank}").into_bytes());
        ep.recv_into(peer, &mut buf, WAIT).unwrap();
        let off = peer * world + rank;
        assert_eq!(buf[0], peer as u8);
        assert!(buf[1..] == big[off..off + MIB4], "4 MiB from {peer} corrupted");
    }
}

#[test]
fn child_process_entry() {
    let Ok(role) = std::env::var(ROLE_ENV) else { return };
    let rank: usize = std::env::var(RANK_ENV).unwrap().parse().unwrap();
    let addrs: Vec<SocketAddr> =
        std::env::var(ADDRS_ENV).unwrap().split(',').map(|a| a.parse().unwrap()).collect();
    match role.as_str() {
        "exchange" => exchange(rank, &addrs),
        "idle" => {
            let _ep = tcp_mesh(rank, &addrs, CONNECT).unwrap();
            std::thread::sleep(Duration::from_secs(3600));
        }
        other => panic!("unknown role {other}"),
    }
}

#[test]
fn three_processes_form_a_mesh_in_any_start_order() {
    let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for order in orders {
        let addrs = free_addrs(3);
        let mut children = Vec::new();
        for rank in order {
            children.push((rank, spawn_child("exchange", rank, &addrs)));
            std::thread::sleep(Duration::from_millis(150));
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        for (rank, child) in &mut children {
            let status = child.wait_until(deadline);
            assert!(status.success(), "start order {order:?}: rank {rank} exited with {status}");
        }
    }
}

#[test]
fn peer_kill_surfaces_on_recv_within_one_second() {
    let addrs = free_addrs(2);
    let mut child = spawn_child("idle", 1, &addrs);
    let ep = tcp_mesh(0, &addrs, CONNECT).unwrap();
    ep.send(1, b"hello").unwrap();
    let start = Instant::now();
    child.0.kill().unwrap();
    let r = ep.recv_timeout(1, Duration::from_secs(10));
    let elapsed = start.elapsed();
    assert_eq!(r, Err(Error::Transport));
    assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    let mut buf = Vec::new();
    assert_eq!(ep.recv_into(1, &mut buf, Duration::from_secs(10)), Err(Error::Transport));
    assert_eq!(ep.send(1, b"after"), Err(Error::Transport));
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn peer_kill_surfaces_on_send_within_one_second() {
    let addrs = free_addrs(2);
    let mut child = spawn_child("idle", 1, &addrs);
    let ep = tcp_mesh(0, &addrs, CONNECT).unwrap();
    let start = Instant::now();
    child.0.kill().unwrap();
    let small = [0u8; 1024];
    loop {
        if ep.send_parts(1, &[&small]).is_err() {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(1), "send still succeeds after kill");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(ep.send(1, &small), Err(Error::Transport));
}

#[test]
fn thousand_4mib_messages_keep_order_and_bytes() {
    const COUNT: usize = 1000;
    let addrs = free_addrs(2);
    let base = Arc::new(pattern(MIB4 + COUNT));
    let (a, b) = (addrs.clone(), Arc::clone(&base));
    let sender = std::thread::spawn(move || {
        let ep = tcp_mesh(0, &a, CONNECT).unwrap();
        for i in 0..COUNT {
            ep.send_parts(1, &[&(i as u64).to_le_bytes(), &b[i..i + MIB4]]).unwrap();
        }
        // Keep the endpoint until the receiver confirms, so nothing depends on drop flushing.
        assert_eq!(ep.recv_timeout(1, WAIT).unwrap(), b"done".to_vec());
    });
    let ep = tcp_mesh(1, &addrs, CONNECT).unwrap();
    let mut buf = Vec::new();
    for i in 0..COUNT {
        ep.recv_into(0, &mut buf, WAIT).unwrap();
        assert_eq!(buf.len(), 8 + MIB4, "message {i} length");
        assert_eq!(buf[..8], (i as u64).to_le_bytes(), "message {i} out of order");
        assert!(buf[8..] == base[i..i + MIB4], "message {i} bytes differ");
    }
    ep.send(0, b"done").unwrap();
    sender.join().unwrap();
}

/// Rank 1 is a raw socket that speaks the handshake and then reads nothing.
fn mesh_with_raw_peer() -> (impl Transport + 'static, TcpStream) {
    let addrs = free_addrs(2);
    let a = addrs.clone();
    let rank0 = std::thread::spawn(move || tcp_mesh(0, &a, CONNECT).unwrap());
    let mut raw = dial(addrs[0]);
    raw.write_all(&hello(2, 1)).unwrap();
    let mut reply = [0u8; 12];
    raw.read_exact(&mut reply).unwrap();
    assert_eq!(reply, hello(2, 0));
    (rank0.join().unwrap(), raw)
}

#[test]
fn send_returns_before_the_peer_reads_and_a_full_queue_blocks() {
    const EXTRA: usize = 40;
    let total = DEFAULT_QUEUE_DEPTH + EXTRA;
    let (ep, mut raw) = mesh_with_raw_peer();
    let base = Arc::new(pattern(MIB4 + total));
    let start = Instant::now();
    for i in 0..DEFAULT_QUEUE_DEPTH {
        ep.send_parts(1, &[&base[i..i + MIB4]]).unwrap();
    }
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(2), "queued sends took {elapsed:?}; peer read nothing");

    // The peer still reads nothing, so the queue and socket buffers fill and a send blocks.
    let done = Arc::new(AtomicBool::new(false));
    let (flag, b) = (Arc::clone(&done), Arc::clone(&base));
    let sender = std::thread::spawn(move || {
        for i in DEFAULT_QUEUE_DEPTH..total {
            ep.send_parts(1, &[&b[i..i + MIB4]]).unwrap();
        }
        flag.store(true, Ordering::SeqCst);
        ep
    });
    std::thread::sleep(Duration::from_secs(1));
    assert!(!done.load(Ordering::SeqCst), "{EXTRA} x 4 MiB sends to a silent peer never blocked");

    for i in 0..total {
        let frame = read_frame(&mut raw);
        assert!(frame == base[i..i + MIB4], "message {i} differs");
    }
    let _ep = sender.join().unwrap();
    assert!(done.load(Ordering::SeqCst));
}

#[test]
fn rank_dials_lower_and_accepts_higher_with_handshake() {
    let addrs = free_addrs(3);
    let listener0 = TcpListener::bind(addrs[0]).unwrap();
    let a = addrs.clone();
    let rank1 = std::thread::spawn(move || tcp_mesh(1, &a, CONNECT).unwrap());

    // Rank 1 dials rank 0 and speaks first.
    let mut from1 = accept(&listener0);
    let mut got = [0u8; 12];
    from1.read_exact(&mut got).unwrap();
    assert_eq!(got, hello(3, 1));
    from1.write_all(&hello(3, 0)).unwrap();

    // Rank 2 dials rank 1; rank 1 accepts and answers.
    let mut to1 = dial(addrs[1]);
    to1.write_all(&hello(3, 2)).unwrap();
    to1.read_exact(&mut got).unwrap();
    assert_eq!(got, hello(3, 1));

    let ep = rank1.join().unwrap();
    assert_eq!((ep.rank(), ep.world()), (1, 3));
    ep.send(0, b"to zero").unwrap();
    ep.send_parts(2, &[b"to ", b"two"]).unwrap();
    assert_eq!(read_frame(&mut from1), b"to zero".to_vec());
    assert_eq!(read_frame(&mut to1), b"to two".to_vec());
    to1.write_all(&3u64.to_le_bytes()).unwrap();
    to1.write_all(b"abc").unwrap();
    assert_eq!(ep.recv_timeout(2, WAIT).unwrap(), b"abc".to_vec());
}

#[test]
fn world_mismatch_is_transport_error() {
    let addrs = free_addrs(3);
    let start = Instant::now();
    let two = addrs[..2].to_vec();
    let rank0 = std::thread::spawn(move || tcp_mesh(0, &two, CONNECT).map(|_| ()));
    let rank1 = tcp_mesh(1, &addrs, CONNECT).map(|_| ());
    assert_eq!(rank1, Err(Error::Transport));
    assert_eq!(rank0.join().unwrap(), Err(Error::Transport));
    assert!(start.elapsed() < Duration::from_secs(10), "mismatch waited for the timeout");
}

#[test]
fn bad_magic_and_bad_rank_are_transport_errors() {
    for bad in [
        {
            let mut h = hello(2, 1);
            h[0] ^= 0xFF;
            h
        },
        hello(2, 0),
        hello(2, 2),
    ] {
        let addrs = free_addrs(2);
        let a = addrs.clone();
        let rank0 = std::thread::spawn(move || tcp_mesh(0, &a, CONNECT).map(|_| ()));
        let mut raw = dial(addrs[0]);
        raw.write_all(&bad).unwrap();
        assert_eq!(rank0.join().unwrap(), Err(Error::Transport), "hello {bad:?}");
    }
}

#[test]
fn bad_arguments_are_transport_errors() {
    let addrs = free_addrs(2);
    let short = Duration::from_millis(100);
    assert!(matches!(tcp_mesh(0, &addrs[..1], short), Err(Error::Transport)));
    assert!(matches!(tcp_mesh(0, &[], short), Err(Error::Transport)));
    assert!(matches!(tcp_mesh(2, &addrs, short), Err(Error::Transport)));
}

#[test]
fn missing_peers_time_out_with_transport_error() {
    let addrs = free_addrs(2);
    let timeout = Duration::from_millis(300);
    for rank in [0, 1] {
        let start = Instant::now();
        assert!(matches!(tcp_mesh(rank, &addrs, timeout), Err(Error::Transport)), "rank {rank}");
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(250), "rank {rank}: {elapsed:?}");
        assert!(elapsed < Duration::from_secs(5), "rank {rank}: {elapsed:?}");
    }
}

#[test]
fn bad_ranks_on_a_tcp_mesh_endpoint_are_transport_errors() {
    let addrs = free_addrs(2);
    let a = addrs.clone();
    let peer = std::thread::spawn(move || tcp_mesh(1, &a, CONNECT).unwrap());
    let ep = tcp_mesh(0, &addrs, CONNECT).unwrap();
    let _peer = peer.join().unwrap();
    let mut buf = Vec::new();
    let short = Duration::from_millis(50);
    assert_eq!(ep.send_parts(0, &[b"self"]), Err(Error::Transport));
    assert_eq!(ep.send_parts(2, &[b"x"]), Err(Error::Transport));
    assert_eq!(ep.recv_into(0, &mut buf, short), Err(Error::Transport));
    assert_eq!(ep.recv_into(2, &mut buf, short), Err(Error::Transport));
    assert_eq!(ep.recv_into(1, &mut buf, short), Err(Error::Transport));
}
