//! TCP mesh on 127.0.0.1. Each message is a frame: a little-endian `u64`
//! payload length followed by the payload bytes.
//!
//! One reader thread per peer drains the socket into a mailbox, so `send`
//! never waits for the receiving application to call `recv_timeout`.

use crate::mailbox::Mailbox;
use linalg_core::{Error, Transport};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

struct Peer {
    writer: Mutex<TcpStream>,
    /// Same socket, kept for `shutdown` on drop.
    control: TcpStream,
    inbox: Arc<Mailbox>,
    reader: Option<JoinHandle<()>>,
}

/// One rank of a localhost TCP mesh.
struct TcpEndpoint {
    rank: usize,
    world: usize,
    /// Indexed by peer rank; `None` at our own rank.
    peers: Vec<Option<Peer>>,
}

fn io_err<T>(r: std::io::Result<T>) -> Result<T, Error> {
    r.map_err(|_| Error::Transport)
}

fn read_frame(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header).ok()?;
    let len = u64::from_le_bytes(header);
    // Grow with the data actually received rather than trusting `len`.
    let mut payload = Vec::new();
    let got = stream.take(len).read_to_end(&mut payload).ok()?;
    (got as u64 == len).then_some(payload)
}

fn spawn_reader(mut stream: TcpStream, inbox: Arc<Mailbox>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while let Some(frame) = read_frame(&mut stream) {
            if inbox.push(frame).is_err() {
                break;
            }
        }
        inbox.close();
    })
}

impl Peer {
    fn new(stream: TcpStream) -> Result<Self, Error> {
        io_err(stream.set_nodelay(true))?;
        let writer = io_err(stream.try_clone())?;
        let control = io_err(stream.try_clone())?;
        let inbox = Arc::new(Mailbox::default());
        let reader = spawn_reader(stream, Arc::clone(&inbox));
        Ok(Peer { writer: Mutex::new(writer), control, inbox, reader: Some(reader) })
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        // Wakes our reader and gives the remote side EOF.
        let _ = self.control.shutdown(Shutdown::Both);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Transport for TcpEndpoint {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world(&self) -> usize {
        self.world
    }

    fn send(&self, dest: usize, bytes: &[u8]) -> Result<(), Error> {
        let peer = self.peers.get(dest).and_then(Option::as_ref).ok_or(Error::Transport)?;
        let mut writer = peer.writer.lock().map_err(|_| Error::Transport)?;
        io_err(writer.write_all(&(bytes.len() as u64).to_le_bytes()))?;
        io_err(writer.write_all(bytes))
    }

    fn recv_timeout(&self, src: usize, timeout: Duration) -> Result<Vec<u8>, Error> {
        let peer = self.peers.get(src).and_then(Option::as_ref).ok_or(Error::Transport)?;
        peer.inbox.pop_timeout(timeout)
    }

    // Not yet implemented by this transport; see the core contract.
    fn send_parts(&self, _dest: usize, _parts: &[&[u8]]) -> Result<(), Error> {
        Err(Error::Transport)
    }

    fn recv_into(&self, _src: usize, _buf: &mut Vec<u8>, _timeout: Duration) -> Result<(), Error> {
        Err(Error::Transport)
    }
}

/// Builds a full mesh of `world` endpoints over 127.0.0.1 with OS-assigned
/// ports. `world < 2` is `Error::Transport`.
pub fn localhost_mesh(world: usize) -> Result<Vec<impl Transport>, Error> {
    if world < 2 {
        return Err(Error::Transport);
    }
    let listeners = (0..world)
        .map(|_| io_err(TcpListener::bind(("127.0.0.1", 0))))
        .collect::<Result<Vec<_>, _>>()?;
    let mut streams: Vec<Vec<Option<TcpStream>>> =
        (0..world).map(|_| (0..world).map(|_| None).collect()).collect();
    let pairs = (0..world).flat_map(|lo| (lo + 1..world).map(move |hi| (lo, hi)));
    for (lo, hi) in pairs {
        // The connection completes in the listener's backlog, so connect
        // then accept from one thread cannot deadlock.
        let dialer = io_err(TcpStream::connect(io_err(listeners[hi].local_addr())?))?;
        let (acceptor, _) = io_err(listeners[hi].accept())?;
        streams[lo][hi] = Some(dialer);
        streams[hi][lo] = Some(acceptor);
    }
    streams
        .into_iter()
        .enumerate()
        .map(|(rank, row)| {
            let peers = row
                .into_iter()
                .map(|s| s.map(Peer::new).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TcpEndpoint { rank, world, peers })
        })
        .collect()
}

/// Messages a sender may queue per peer before `send` blocks.
pub const DEFAULT_QUEUE_DEPTH: usize = 4;

/// First four bytes of the `tcp_mesh` handshake.
pub const HANDSHAKE_MAGIC: [u8; 4] = *b"LNGM";

/// Not yet implemented.
pub fn tcp_mesh(
    _rank: usize,
    _addrs: &[std::net::SocketAddr],
    _connect_timeout: Duration,
) -> Result<impl Transport, Error> {
    Err::<TcpEndpoint, _>(Error::Transport)
}
