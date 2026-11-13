//! `tcp_mesh`: one endpoint per process, connected to every other rank by address.
//!
//! Rank `r` listens on `addrs[r]`, dials every lower rank in order, then accepts every higher
//! rank in any order. Each connection opens with a handshake: the dialer sends
//! `(HANDSHAKE_MAGIC, world, rank)` as 4 bytes and two little-endian `u32`s, and the acceptor
//! checks it and answers with its own. Anything unexpected is `Error::Transport`.
//!
//! Every rank binds before it dials, and a rank accepts only after its own dials finish, which
//! needs only lower ranks; so the mesh forms whatever order the processes start in.

use crate::tcp::{io_err, TcpEndpoint, DEFAULT_QUEUE_DEPTH};
use linalg_core::{Error, Transport};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// First four bytes of the `tcp_mesh` handshake.
pub const HANDSHAKE_MAGIC: [u8; 4] = *b"LNGM";

const HELLO_LEN: usize = 12;

/// Pause between attempts while a peer is not yet listening or dialing.
const RETRY: Duration = Duration::from_millis(5);

/// Builds rank `rank`'s endpoint of a mesh whose rank `r` listens on `addrs[r]`, with
/// [`DEFAULT_QUEUE_DEPTH`] messages queued per peer before `send` blocks.
///
/// Every peer must connect and complete the handshake within `connect_timeout` of this call;
/// until then dialing retries. Fewer than two addresses, `rank` out of range, a bind failure,
/// a handshake mismatch, or the timeout is `Error::Transport`.
pub fn tcp_mesh(
    rank: usize,
    addrs: &[SocketAddr],
    connect_timeout: Duration,
) -> Result<impl Transport, Error> {
    tcp_mesh_with_depth(rank, addrs, connect_timeout, DEFAULT_QUEUE_DEPTH)
}

/// [`tcp_mesh`] with `queue_depth` messages queued per peer before `send` blocks. A depth of 0
/// is `Error::Transport`.
pub fn tcp_mesh_with_depth(
    rank: usize,
    addrs: &[SocketAddr],
    connect_timeout: Duration,
    queue_depth: usize,
) -> Result<impl Transport, Error> {
    let world = addrs.len();
    if world < 2 || rank >= world || u32::try_from(world).is_err() || queue_depth == 0 {
        return Err(Error::Transport);
    }
    let deadline = Deadline::after(connect_timeout);
    let listener = io_err(TcpListener::bind(addrs[rank]))?;
    let mut streams: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();

    for (lower, addr) in addrs.iter().enumerate().take(rank) {
        let mut stream = dial(*addr, &deadline)?;
        handshake_io(&stream, &deadline)?;
        io_err(stream.write_all(&hello(world, rank)))?;
        if read_hello(&mut stream)? != (world, lower) {
            return Err(Error::Transport);
        }
        streams[lower] = Some(stream);
    }

    io_err(listener.set_nonblocking(true))?;
    for _ in rank + 1..world {
        let mut stream = accept(&listener, &deadline)?;
        handshake_io(&stream, &deadline)?;
        let (their_world, their_rank) = read_hello(&mut stream)?;
        if their_world != world
            || their_rank <= rank
            || their_rank >= world
            || streams[their_rank].is_some()
        {
            return Err(Error::Transport);
        }
        io_err(stream.write_all(&hello(world, rank)))?;
        streams[their_rank] = Some(stream);
    }

    for stream in streams.iter().flatten() {
        io_err(stream.set_read_timeout(None))?;
        io_err(stream.set_write_timeout(None))?;
    }
    TcpEndpoint::new(rank, streams, queue_depth)
}

struct Deadline(Option<Instant>);

impl Deadline {
    fn after(timeout: Duration) -> Self {
        Deadline(Instant::now().checked_add(timeout))
    }

    /// Time left, or `Error::Transport` once it has passed. `None` inside means no deadline.
    fn remaining(&self) -> Result<Option<Duration>, Error> {
        match self.0 {
            None => Ok(None),
            Some(at) => {
                let left = at.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    Err(Error::Transport)
                } else {
                    Ok(Some(left))
                }
            }
        }
    }
}

fn hello(world: usize, rank: usize) -> [u8; HELLO_LEN] {
    let mut out = [0u8; HELLO_LEN];
    out[0..4].copy_from_slice(&HANDSHAKE_MAGIC);
    out[4..8].copy_from_slice(&(world as u32).to_le_bytes());
    out[8..12].copy_from_slice(&(rank as u32).to_le_bytes());
    out
}

/// Returns `(world, rank)` from a peer's hello; a wrong magic is `Error::Transport`.
fn read_hello(stream: &mut TcpStream) -> Result<(usize, usize), Error> {
    let mut bytes = [0u8; HELLO_LEN];
    io_err(stream.read_exact(&mut bytes))?;
    if bytes[0..4] != HANDSHAKE_MAGIC {
        return Err(Error::Transport);
    }
    let field = |at: usize| {
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[at..at + 4]);
        u32::from_le_bytes(word) as usize
    };
    Ok((field(4), field(8)))
}

/// Bounds the handshake's reads and writes by the deadline.
fn handshake_io(stream: &TcpStream, deadline: &Deadline) -> Result<(), Error> {
    let left = deadline.remaining()?;
    io_err(stream.set_nonblocking(false))?;
    io_err(stream.set_read_timeout(left))?;
    io_err(stream.set_write_timeout(left))
}

fn dial(addr: SocketAddr, deadline: &Deadline) -> Result<TcpStream, Error> {
    loop {
        let attempt = match deadline.remaining()? {
            Some(left) => TcpStream::connect_timeout(&addr, left),
            None => TcpStream::connect(addr),
        };
        match attempt {
            // A dial to a local port nobody listens on yet can connect to itself (TCP
            // simultaneous open); that is not the peer, so try again.
            Ok(stream) if is_self_connected(&stream) => std::thread::sleep(RETRY),
            Ok(stream) => return Ok(stream),
            Err(_) => std::thread::sleep(RETRY),
        }
    }
}

fn is_self_connected(stream: &TcpStream) -> bool {
    matches!((stream.local_addr(), stream.peer_addr()), (Ok(local), Ok(peer)) if local == peer)
}

fn accept(listener: &TcpListener, deadline: &Deadline) -> Result<TcpStream, Error> {
    loop {
        deadline.remaining()?;
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::WouldBlock | ErrorKind::Interrupted | ErrorKind::ConnectionAborted
                ) =>
            {
                std::thread::sleep(RETRY)
            }
            Err(_) => return Err(Error::Transport),
        }
    }
}
