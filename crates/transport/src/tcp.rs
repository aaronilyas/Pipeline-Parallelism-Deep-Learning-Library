//! TCP endpoints. Each message is a frame: a little-endian `u64` payload length followed by
//! the payload bytes.
//!
//! Per peer, a writer thread drains a bounded [`Outbox`] onto the socket with vectored writes,
//! and a reader thread reads frames ahead into recycled buffers in an [`Inbox`]. So `send`
//! copies once and returns, and `recv_into` swaps a ready buffer in without copying.
//!
//! A write error, or end of stream or a read error from the peer, fails the outbox (every later
//! send is `Error::Transport`) and closes the inbox (queued messages are still delivered, then
//! every recv is `Error::Transport`).

use crate::queue::{Inbox, Outbox};
use crate::sockbuf;
use linalg_core::{Error, Transport};
use std::io::{ErrorKind, IoSlice, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Messages a sender may queue per peer before `send` blocks.
pub const DEFAULT_QUEUE_DEPTH: usize = 4;

/// Bytes the reader allocates ahead of the data actually received, so a corrupt length
/// cannot make it allocate more than the peer sends.
const GROW_STEP: usize = 1 << 20;

/// How long `Drop` waits for the reader to see end of stream before detaching it.
const READER_GRACE: Duration = Duration::from_secs(1);

struct Peer {
    outbox: Arc<Outbox>,
    inbox: Arc<Inbox>,
    /// Same socket, kept for `shutdown`.
    control: TcpStream,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
}

/// One rank of a TCP mesh.
pub(crate) struct TcpEndpoint {
    rank: usize,
    world: usize,
    /// Indexed by peer rank; `None` at our own rank.
    peers: Vec<Option<Peer>>,
}

pub(crate) fn io_err<T>(r: std::io::Result<T>) -> Result<T, Error> {
    r.map_err(|_| Error::Transport)
}

/// Writes the frame header and `message` with vectored writes until both are on the socket.
fn write_frame(stream: &mut TcpStream, message: &[u8]) -> std::io::Result<()> {
    let header = (message.len() as u64).to_le_bytes();
    let mut slices = [IoSlice::new(&header), IoSlice::new(message)];
    let mut remaining = &mut slices[..];
    let mut left = header.len() + message.len();
    while left > 0 {
        match stream.write_vectored(remaining) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => {
                left -= n;
                IoSlice::advance_slices(&mut remaining, n);
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Reads exactly `len` bytes into `buf`, reusing its initialized length so that a recycled
/// buffer is neither zeroed nor reallocated.
fn read_payload(stream: &mut TcpStream, buf: &mut Vec<u8>, len: usize) -> std::io::Result<()> {
    if buf.len() >= len {
        buf.truncate(len);
        return stream.read_exact(buf);
    }
    let mut filled = buf.len();
    stream.read_exact(buf)?;
    while filled < len {
        let step = (len - filled).min(filled.max(GROW_STEP));
        buf.resize(filled + step, 0);
        stream.read_exact(&mut buf[filled..])?;
        filled += step;
    }
    Ok(())
}

fn spawn_writer(mut stream: TcpStream, outbox: Arc<Outbox>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while let Some(message) = outbox.pop() {
            if write_frame(&mut stream, &message).is_err() {
                outbox.fail();
                // Wakes the reader, which closes the inbox once queued messages are taken.
                let _ = stream.shutdown(Shutdown::Both);
                return;
            }
            outbox.recycle(message);
        }
    })
}

fn spawn_reader(mut stream: TcpStream, inbox: Arc<Inbox>, outbox: Arc<Outbox>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        loop {
            let mut header = [0u8; 8];
            if stream.read_exact(&mut header).is_err() {
                break;
            }
            let Ok(len) = usize::try_from(u64::from_le_bytes(header)) else { break };
            let mut buf = inbox.take_free();
            if read_payload(&mut stream, &mut buf, len).is_err() {
                break;
            }
            inbox.push(buf);
        }
        // The peer is gone (or we are shutting down): later sends fail rather than queue.
        outbox.fail();
        // Answers the peer's end of stream with ours. Windows does not wake a reader blocked
        // in `recv` when its own side shuts down, so a dropping peer waits for this.
        let _ = stream.shutdown(Shutdown::Both);
        inbox.close();
    })
}

impl Peer {
    fn new(stream: TcpStream, depth: usize) -> Result<Self, Error> {
        io_err(stream.set_nodelay(true))?;
        sockbuf::tune(&stream);
        let writer_stream = io_err(stream.try_clone())?;
        let control = io_err(stream.try_clone())?;
        let outbox = Arc::new(Outbox::new(depth));
        let inbox = Arc::new(Inbox::new());
        let writer = spawn_writer(writer_stream, Arc::clone(&outbox));
        let reader = spawn_reader(stream, Arc::clone(&inbox), Arc::clone(&outbox));
        Ok(Peer { outbox, inbox, control, writer: Some(writer), reader: Some(reader) })
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        // Queued messages go out first, as they would have with synchronous writes.
        self.outbox.close();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        // Gives the remote side end of stream, and wakes our reader on Linux and macOS. On
        // Windows the reader wakes when the peer answers with its own end of stream; a peer
        // that never does leaves the reader detached, holding the socket until the OS gives up.
        let _ = self.control.shutdown(Shutdown::Both);
        if let Some(reader) = self.reader.take() {
            if self.inbox.wait_closed(READER_GRACE) {
                let _ = reader.join();
            }
        }
    }
}

impl TcpEndpoint {
    /// `streams[r]` is the connection to rank `r`, `None` at `rank`.
    pub(crate) fn new(
        rank: usize,
        streams: Vec<Option<TcpStream>>,
        depth: usize,
    ) -> Result<Self, Error> {
        let world = streams.len();
        let peers = streams
            .into_iter()
            .map(|s| s.map(|s| Peer::new(s, depth)).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TcpEndpoint { rank, world, peers })
    }

    fn peer(&self, rank: usize) -> Result<&Peer, Error> {
        self.peers.get(rank).and_then(Option::as_ref).ok_or(Error::Transport)
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
        self.peer(dest)?.outbox.push_parts(&[bytes])
    }

    fn recv_timeout(&self, src: usize, timeout: Duration) -> Result<Vec<u8>, Error> {
        self.peer(src)?.inbox.pop_timeout(timeout)
    }

    fn send_parts(&self, dest: usize, parts: &[&[u8]]) -> Result<(), Error> {
        self.peer(dest)?.outbox.push_parts(parts)
    }

    fn recv_into(&self, src: usize, buf: &mut Vec<u8>, timeout: Duration) -> Result<(), Error> {
        self.peer(src)?.inbox.pop_into(buf, timeout)
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
        .map(|(rank, row)| TcpEndpoint::new(rank, row, DEFAULT_QUEUE_DEPTH))
        .collect()
}
