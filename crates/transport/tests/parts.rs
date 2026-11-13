//! `send_parts` and `recv_into` on the existing constructors, `loopback` and `localhost_mesh`.

use linalg_core::{Error, Transport};
use linalg_transport::{localhost_mesh, loopback};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);

fn parts_equal_concatenation_and_recv_into_reuses_buf<T: Transport>(eps: Vec<T>) {
    let (a, b) = (&eps[0], &eps[1]);
    a.send_parts(1, &[b"head", b"", b"er", &[0xFF, 0x00]]).unwrap();
    a.send_parts(1, &[]).unwrap();
    a.send(1, b"plain").unwrap();
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), b"header\xFF\x00".to_vec());
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(b"stale contents");
    b.recv_into(0, &mut buf, WAIT).unwrap();
    assert!(buf.is_empty());
    b.recv_into(0, &mut buf, WAIT).unwrap();
    assert_eq!(buf, b"plain".to_vec());
    let short = Duration::from_millis(50);
    assert_eq!(b.recv_into(0, &mut buf, short), Err(Error::Transport));
    assert_eq!(a.send_parts(0, &[b"self"]), Err(Error::Transport));
    assert_eq!(a.send_parts(2, &[b"x"]), Err(Error::Transport));
    assert_eq!(a.recv_into(2, &mut buf, short), Err(Error::Transport));
}

#[test]
fn loopback_send_parts_and_recv_into() {
    parts_equal_concatenation_and_recv_into_reuses_buf(loopback(2).unwrap());
}

#[test]
fn localhost_mesh_send_parts_and_recv_into() {
    parts_equal_concatenation_and_recv_into_reuses_buf(localhost_mesh(2).unwrap());
}

#[test]
fn loopback_recv_into_keeps_the_callers_allocation() {
    let eps = loopback(2).unwrap();
    let mut buf = Vec::with_capacity(1024);
    let ptr = buf.as_ptr();
    for i in 0..100u32 {
        eps[0].send_parts(1, &[&i.to_le_bytes(), &[7u8; 500]]).unwrap();
        eps[1].recv_into(0, &mut buf, WAIT).unwrap();
        assert_eq!(buf[..4], i.to_le_bytes());
        assert_eq!(buf.as_ptr(), ptr);
    }
}
