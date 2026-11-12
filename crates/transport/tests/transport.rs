use linalg_core::{Error, Transport};
use linalg_transport::{localhost_mesh, loopback};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(10);

fn send_copies_and_recv_allocation_is_distinct<T: Transport>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let a = eps.remove(0);
    let mut src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    a.send(1, &src).unwrap();
    let src_ptr = src.as_ptr();
    src.fill(0xEE);
    let got = b.recv_timeout(0, WAIT).unwrap();
    assert_eq!(got, vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
    assert_ne!(got.as_ptr(), src_ptr);
}

fn two_messages_preserve_order_and_raw_bytes<T: Transport>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let a = eps.remove(0);
    let first: Vec<u8> = vec![0xFF, 0x00, 0xFE, 0x80, 0xC0, 0x01];
    // Looks like a framed message: a little-endian length of 3 plus "abc".
    let mut second = 3u64.to_le_bytes().to_vec();
    second.extend_from_slice(b"abc");
    // Looks like nothing but a length.
    let third = 1_000_000u64.to_le_bytes().to_vec();
    a.send(1, &first).unwrap();
    a.send(1, &second).unwrap();
    a.send(1, &third).unwrap();
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), first);
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), second);
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), third);
}

fn empty_payload_and_non_utf8<T: Transport>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let a = eps.remove(0);
    let bad = vec![0xC3u8, 0x28, 0xA0, 0xA1, 0xE2, 0x28, 0xA1];
    assert!(std::str::from_utf8(&bad).is_err());
    a.send(1, &[]).unwrap();
    a.send(1, &bad).unwrap();
    a.send(1, &[]).unwrap();
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), Vec::<u8>::new());
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), bad.clone());
    assert_eq!(b.recv_timeout(0, WAIT).unwrap(), Vec::<u8>::new());
}

fn bad_rank_is_transport_error<T: Transport>(eps: Vec<T>) {
    let world = eps.len();
    let a = &eps[0];
    assert_eq!(a.world(), world);
    assert_eq!(a.send(0, b"self"), Err(Error::Transport));
    assert_eq!(a.send(world, b"x"), Err(Error::Transport));
    assert_eq!(a.send(usize::MAX, b"x"), Err(Error::Transport));
    assert_eq!(a.recv_timeout(world, Duration::from_millis(50)), Err(Error::Transport));
    assert_eq!(a.recv_timeout(usize::MAX, Duration::from_millis(50)), Err(Error::Transport));
    assert_eq!(a.recv_timeout(0, Duration::from_millis(50)), Err(Error::Transport));
    // Bad sends queue nothing.
    assert_eq!(eps[1].recv_timeout(0, Duration::from_millis(50)), Err(Error::Transport));
}

fn timeout_returns_in_under_two_seconds<T: Transport>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let start = Instant::now();
    let r = b.recv_timeout(0, Duration::from_millis(200));
    let elapsed = start.elapsed();
    assert_eq!(r, Err(Error::Transport));
    assert!(elapsed >= Duration::from_millis(150), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
}

fn mesh_of_three_ranks_delivers_from_zero_to_two_while_one_is_idle<T: Transport>(eps: Vec<T>) {
    assert_eq!(eps.len(), 3);
    for (i, e) in eps.iter().enumerate() {
        assert_eq!(e.rank(), i);
        assert_eq!(e.world(), 3);
    }
    eps[0].send(2, b"zero to two").unwrap();
    assert_eq!(eps[2].recv_timeout(0, WAIT).unwrap(), b"zero to two".to_vec());
    // Rank 1 is idle: nothing arrives from 0, and nothing from 1 reaches 2.
    let short = Duration::from_millis(100);
    assert_eq!(eps[1].recv_timeout(0, short), Err(Error::Transport));
    assert_eq!(eps[2].recv_timeout(1, short), Err(Error::Transport));
    assert_eq!(eps[2].recv_timeout(0, short), Err(Error::Transport));
}

fn different_sources_never_merge<T: Transport>(eps: Vec<T>) {
    eps[0].send(2, b"from zero").unwrap();
    eps[1].send(2, b"from one").unwrap();
    eps[0].send(2, b"zero again").unwrap();
    assert_eq!(eps[2].recv_timeout(1, WAIT).unwrap(), b"from one".to_vec());
    assert_eq!(eps[2].recv_timeout(0, WAIT).unwrap(), b"from zero".to_vec());
    assert_eq!(eps[2].recv_timeout(0, WAIT).unwrap(), b"zero again".to_vec());
}

fn send_does_not_wait_for_receiver<T: Transport>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let a = eps.remove(0);
    let big: Vec<u8> = (0..8 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    // Neither side has called recv yet; send must still return.
    a.send(1, &big).unwrap();
    for i in 0..200u32 {
        a.send(1, &i.to_le_bytes()).unwrap();
    }
    assert!(b.recv_timeout(0, WAIT).unwrap() == big, "large payload corrupted or missing");
    for i in 0..200u32 {
        assert_eq!(b.recv_timeout(0, WAIT).unwrap(), i.to_le_bytes().to_vec());
    }
}

fn endpoints_move_to_threads<T: Transport + 'static>(mut eps: Vec<T>) {
    let b = eps.pop().unwrap();
    let a = eps.remove(0);
    let ta = std::thread::spawn(move || {
        a.send(1, b"ping").unwrap();
        a.recv_timeout(1, WAIT).unwrap()
    });
    let tb = std::thread::spawn(move || {
        let got = b.recv_timeout(0, WAIT).unwrap();
        b.send(0, b"pong").unwrap();
        got
    });
    assert_eq!(tb.join().unwrap(), b"ping".to_vec());
    assert_eq!(ta.join().unwrap(), b"pong".to_vec());
}

macro_rules! invariant_tests {
    ($modname:ident, $make:expr) => {
        mod $modname {
            use super::*;

            #[test]
            fn send_copies_and_recv_allocation_is_distinct() {
                super::send_copies_and_recv_allocation_is_distinct($make(2));
            }
            #[test]
            fn two_messages_preserve_order_and_raw_bytes() {
                super::two_messages_preserve_order_and_raw_bytes($make(2));
            }
            #[test]
            fn empty_payload_and_non_utf8() {
                super::empty_payload_and_non_utf8($make(2));
            }
            #[test]
            fn bad_rank_is_transport_error() {
                super::bad_rank_is_transport_error($make(3));
            }
            #[test]
            fn timeout_returns_in_under_two_seconds() {
                super::timeout_returns_in_under_two_seconds($make(2));
            }
            #[test]
            fn mesh_of_three_ranks_delivers_from_zero_to_two_while_one_is_idle() {
                super::mesh_of_three_ranks_delivers_from_zero_to_two_while_one_is_idle($make(3));
            }
            #[test]
            fn different_sources_never_merge() {
                super::different_sources_never_merge($make(3));
            }
            #[test]
            fn send_does_not_wait_for_receiver() {
                super::send_does_not_wait_for_receiver($make(2));
            }
            #[test]
            fn endpoints_move_to_threads() {
                super::endpoints_move_to_threads($make(2));
            }
            #[test]
            fn world_below_two_is_transport_error() {
                for w in [0usize, 1] {
                    assert!(matches!(make_result(w), Err(Error::Transport)));
                }
            }
        }
    };
}

fn make_loopback(world: usize) -> Vec<impl Transport + 'static> {
    loopback(world).unwrap()
}

fn make_tcp(world: usize) -> Vec<impl Transport + 'static> {
    localhost_mesh(world).unwrap()
}

mod loopback_suite {
    use super::*;

    fn make_result(world: usize) -> Result<usize, Error> {
        loopback(world).map(|v| v.len())
    }

    invariant_tests!(invariants, super::super::make_loopback);

    #[test]
    fn mailbox_holds_a_copy_after_source_is_dropped() {
        let eps = loopback(2).unwrap();
        {
            let src = vec![9u8; 64];
            eps[0].send(1, &src).unwrap();
        }
        assert_eq!(eps[1].recv_timeout(0, WAIT).unwrap(), vec![9u8; 64]);
    }
}

mod tcp_suite {
    use super::*;

    fn make_result(world: usize) -> Result<usize, Error> {
        localhost_mesh(world).map(|v| v.len())
    }

    invariant_tests!(invariants, super::super::make_tcp);

    #[test]
    fn dropping_an_endpoint_before_recv_ends_it_with_transport_error() {
        let mut eps = localhost_mesh(2).unwrap();
        let b = eps.pop().unwrap();
        let a = eps.remove(0);
        drop(a);
        let start = Instant::now();
        let r = b.recv_timeout(0, Duration::from_secs(20));
        assert_eq!(r, Err(Error::Transport));
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn dropping_an_endpoint_ends_a_pending_recv_before_its_timeout() {
        let mut eps = localhost_mesh(2).unwrap();
        let b = eps.pop().unwrap();
        let a = eps.remove(0);
        let start = Instant::now();
        let waiter = std::thread::spawn(move || b.recv_timeout(0, Duration::from_secs(20)));
        drop(a);
        assert_eq!(waiter.join().unwrap(), Err(Error::Transport));
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn queued_messages_survive_the_sender_dropping() {
        let mut eps = localhost_mesh(2).unwrap();
        let b = eps.pop().unwrap();
        let a = eps.remove(0);
        a.send(1, b"last words").unwrap();
        drop(a);
        assert_eq!(b.recv_timeout(0, WAIT).unwrap(), b"last words".to_vec());
        assert_eq!(b.recv_timeout(0, WAIT), Err(Error::Transport));
    }

    #[test]
    fn send_to_a_dropped_peer_eventually_errors_without_panicking() {
        let mut eps = localhost_mesh(2).unwrap();
        let b = eps.pop().unwrap();
        let a = eps.remove(0);
        drop(b);
        let big = vec![0u8; 1 << 20];
        let mut failed = false;
        for _ in 0..64 {
            if a.send(1, &big).is_err() {
                failed = true;
                break;
            }
        }
        assert!(failed);
    }
}
