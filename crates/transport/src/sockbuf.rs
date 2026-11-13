//! Socket send and receive buffers of at least [`SOCKET_BUFFER`] where the OS allows.
//!
//! `std` has no setter for `SO_SNDBUF` / `SO_RCVBUF`, so this calls the C library's
//! `setsockopt` (Linux, macOS) or Winsock's (Windows), both of which `std` already links.
//!
//! On Linux an explicit size is capped at `net.core.{w,r}mem_max` and switches kernel
//! autotuning off, so it is set only when that cap is at least [`SOCKET_BUFFER`]; otherwise
//! autotuning, bounded by `net.ipv4.tcp_{w,r}mem`, is left in charge. Failures are ignored:
//! the buffers are a performance hint, not a requirement of the protocol.

use std::net::TcpStream;

/// Requested size of each socket buffer.
pub(crate) const SOCKET_BUFFER: usize = 4 << 20;

#[derive(Clone, Copy)]
pub(crate) enum Which {
    Send,
    Recv,
}

pub(crate) fn tune(stream: &TcpStream) {
    for which in [Which::Send, Which::Recv] {
        if allowed(which) && get(stream, which).is_some_and(|size| size < SOCKET_BUFFER) {
            set(stream, which, SOCKET_BUFFER);
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn allowed(which: Which) -> bool {
    let path = match which {
        Which::Send => "/proc/sys/net/core/wmem_max",
        Which::Recv => "/proc/sys/net/core/rmem_max",
    };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse::<usize>().ok())
        .is_some_and(|max| max >= SOCKET_BUFFER)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn allowed(_which: Which) -> bool {
    true
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
mod sys {
    use super::Which;
    use std::net::TcpStream;

    #[cfg(target_os = "linux")]
    const SOL_SOCKET: i32 = 1;
    #[cfg(target_os = "linux")]
    fn name(which: Which) -> i32 {
        match which {
            Which::Send => 7, // SO_SNDBUF
            Which::Recv => 8, // SO_RCVBUF
        }
    }

    #[cfg(any(target_os = "macos", windows))]
    const SOL_SOCKET: i32 = 0xffff;
    #[cfg(any(target_os = "macos", windows))]
    fn name(which: Which) -> i32 {
        match which {
            Which::Send => 0x1001, // SO_SNDBUF
            Which::Recv => 0x1002, // SO_RCVBUF
        }
    }

    #[cfg(unix)]
    mod ffi {
        extern "C" {
            pub fn setsockopt(fd: i32, level: i32, name: i32, value: *const u8, len: u32) -> i32;
            pub fn getsockopt(fd: i32, level: i32, name: i32, value: *mut u8, len: *mut u32)
                -> i32;
        }
    }

    #[cfg(unix)]
    pub(crate) fn get(stream: &TcpStream, which: Which) -> Option<usize> {
        use std::os::fd::AsRawFd;
        let mut value: i32 = 0;
        let mut len = std::mem::size_of::<i32>() as u32;
        // SAFETY: `value` and `len` are live, writable, and `len` holds `value`'s size.
        let rc = unsafe {
            ffi::getsockopt(
                stream.as_raw_fd(),
                SOL_SOCKET,
                name(which),
                (&mut value as *mut i32).cast(),
                &mut len,
            )
        };
        (rc == 0).then(|| usize::try_from(value).unwrap_or(0))
    }

    #[cfg(unix)]
    pub(crate) fn set(stream: &TcpStream, which: Which, size: usize) {
        use std::os::fd::AsRawFd;
        let value = i32::try_from(size).unwrap_or(i32::MAX);
        // SAFETY: `value` is live for the call and the length passed is its size.
        unsafe {
            ffi::setsockopt(
                stream.as_raw_fd(),
                SOL_SOCKET,
                name(which),
                (&value as *const i32).cast(),
                std::mem::size_of::<i32>() as u32,
            );
        }
    }

    #[cfg(windows)]
    mod ffi {
        #[link(name = "ws2_32")]
        extern "system" {
            pub fn setsockopt(s: usize, level: i32, name: i32, value: *const u8, len: i32) -> i32;
            pub fn getsockopt(
                s: usize,
                level: i32,
                name: i32,
                value: *mut u8,
                len: *mut i32,
            ) -> i32;
        }
    }

    #[cfg(windows)]
    pub(crate) fn get(stream: &TcpStream, which: Which) -> Option<usize> {
        use std::os::windows::io::AsRawSocket;
        let mut value: i32 = 0;
        let mut len = std::mem::size_of::<i32>() as i32;
        // SAFETY: `value` and `len` are live, writable, and `len` holds `value`'s size.
        let rc = unsafe {
            ffi::getsockopt(
                stream.as_raw_socket() as usize,
                SOL_SOCKET,
                name(which),
                (&mut value as *mut i32).cast(),
                &mut len,
            )
        };
        (rc == 0).then(|| usize::try_from(value).unwrap_or(0))
    }

    #[cfg(windows)]
    pub(crate) fn set(stream: &TcpStream, which: Which, size: usize) {
        use std::os::windows::io::AsRawSocket;
        let value = i32::try_from(size).unwrap_or(i32::MAX);
        // SAFETY: `value` is live for the call and the length passed is its size.
        unsafe {
            ffi::setsockopt(
                stream.as_raw_socket() as usize,
                SOL_SOCKET,
                name(which),
                (&value as *const i32).cast(),
                std::mem::size_of::<i32>() as i32,
            );
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
pub(crate) use sys::{get, set};

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(crate) fn get(_stream: &TcpStream, _which: Which) -> Option<usize> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(crate) fn set(_stream: &TcpStream, _which: Which, _size: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn tuned_buffers_are_at_least_4_mib_where_allowed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let dialer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (acceptor, _) = listener.accept().unwrap();
        for stream in [&dialer, &acceptor] {
            tune(stream);
            for which in [Which::Send, Which::Recv] {
                let size = get(stream, which).expect("getsockopt");
                if allowed(which) {
                    assert!(size >= SOCKET_BUFFER, "buffer is {size} bytes");
                }
            }
        }
    }
}
