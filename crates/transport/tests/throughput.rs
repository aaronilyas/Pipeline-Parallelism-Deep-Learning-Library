//! Release only: 4 MiB messages through `tcp_mesh` over 127.0.0.1 reach at least 80% of a raw
//! `TcpStream` `write_all` / `read_exact` loop measured in the same test.
#![cfg(not(debug_assertions))]

use linalg_core::Transport;
use linalg_transport::tcp_mesh;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

const MIB4: usize = 4 << 20;
const MESSAGES: usize = 256;
const TRIALS: usize = 5;
const WAIT: Duration = Duration::from_secs(30);

fn raw_bytes_per_sec() -> f64 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let reader = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_nodelay(true).unwrap();
        let mut buf = vec![0u8; MIB4];
        for _ in 0..MESSAGES {
            s.read_exact(&mut buf).unwrap();
        }
        Instant::now()
    });
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_nodelay(true).unwrap();
    let payload = vec![7u8; MIB4];
    let start = Instant::now();
    for _ in 0..MESSAGES {
        s.write_all(&payload).unwrap();
    }
    let end = reader.join().unwrap();
    (MESSAGES * MIB4) as f64 / (end - start).as_secs_f64()
}

fn mesh_bytes_per_sec() -> f64 {
    let listeners: Vec<TcpListener> =
        (0..2).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    drop(listeners);
    let a = addrs.clone();
    let receiver = std::thread::spawn(move || {
        let ep = tcp_mesh(1, &a, WAIT).unwrap();
        let mut buf = Vec::with_capacity(MIB4);
        for _ in 0..MESSAGES {
            ep.recv_into(0, &mut buf, WAIT).unwrap();
            assert_eq!(buf.len(), MIB4);
        }
        let end = Instant::now();
        ep.send(0, b"done").unwrap();
        end
    });
    let ep = tcp_mesh(0, &addrs, WAIT).unwrap();
    let payload = vec![7u8; MIB4];
    let start = Instant::now();
    for _ in 0..MESSAGES {
        ep.send_parts(1, &[&payload]).unwrap();
    }
    let end = receiver.join().unwrap();
    assert_eq!(ep.recv_timeout(1, WAIT).unwrap(), b"done".to_vec());
    (MESSAGES * MIB4) as f64 / (end - start).as_secs_f64()
}

#[test]
fn tcp_mesh_reaches_80_percent_of_raw_tcp_on_localhost() {
    let (mut raw, mut mesh) = (0f64, 0f64);
    for _ in 0..TRIALS {
        raw = raw.max(raw_bytes_per_sec());
        mesh = mesh.max(mesh_bytes_per_sec());
    }
    let ratio = mesh / raw;
    eprintln!(
        "throughput: raw {:.3} GB/s, tcp_mesh {:.3} GB/s, ratio {ratio:.3} \
         ({MESSAGES} x 4 MiB, best of {TRIALS})",
        raw / 1e9,
        mesh / 1e9
    );
    assert!(ratio >= 0.8, "tcp_mesh {mesh:.0} B/s is below 80% of raw {raw:.0} B/s");
}
