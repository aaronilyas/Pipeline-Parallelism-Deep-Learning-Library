//! `write_wire_header` and `parse_wire_header` allocate nothing on the heap except the shape
//! `parse_wire_header` returns. Counted per thread, so parallel tests do not interfere.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use linalg_core::{encode_wire, parse_wire_header, write_wire_header, DType, WIRE_HEADER_MAX};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs_during<T>(f: impl FnOnce() -> T) -> (usize, T) {
    let before = ALLOCS.with(Cell::get);
    let out = f();
    (ALLOCS.with(Cell::get) - before, out)
}

#[test]
fn write_header_never_allocates() {
    let shapes: [&[usize]; 4] = [&[], &[4096], &[256, 4096], &[1, 1, 1, 1, 1, 1, 1, 1]];
    let mut out = [0u8; WIRE_HEADER_MAX];
    for shape in shapes {
        let (n, len) = allocs_during(|| write_wire_header(DType::F32, shape, &mut out));
        assert_eq!(n, 0, "{shape:?}");
        assert_eq!(len, Ok(16 + 4 * shape.len()));
    }
    let (n, err) = allocs_during(|| write_wire_header(DType::F32, &[1; 9], &mut out));
    assert_eq!(n, 0);
    assert!(err.is_err());
    let (n, err) = allocs_during(|| write_wire_header(DType::F32, &[1 << 31, 1 << 31], &mut out));
    assert_eq!(n, 0);
    assert!(err.is_err());
}

#[test]
fn parse_header_allocates_only_the_returned_shape() {
    let scalar = encode_wire(DType::BF16, &[], &[1, 2]).unwrap();
    let (n, parsed) = allocs_during(|| parse_wire_header(&scalar));
    assert_eq!(parsed.map(|p| p.1.capacity()), Ok(0));
    assert_eq!(n, 0, "rank 0 has an empty shape");

    let matrix = encode_wire(DType::F32, &[2, 3], &[0; 24]).unwrap();
    let (n, parsed) = allocs_during(|| parse_wire_header(&matrix));
    assert_eq!(parsed, Ok((DType::F32, vec![2, 3], 24)));
    assert_eq!(n, 1, "one allocation: the shape");

    let mut trailing = matrix.clone();
    trailing.push(0);
    let (n, parsed) = allocs_during(|| parse_wire_header(&trailing));
    assert!(parsed.is_err());
    assert_eq!(n, 0, "a rejected message allocates nothing");

    let mut overflow = encode_wire(DType::F32, &[1, 1], &[0; 4]).unwrap();
    overflow[16..20].copy_from_slice(&(1u32 << 31).to_le_bytes());
    overflow[20..24].copy_from_slice(&(1u32 << 31).to_le_bytes());
    let (n, parsed) = allocs_during(|| parse_wire_header(&overflow));
    assert!(parsed.is_err());
    assert_eq!(n, 0, "a rejected message allocates nothing");
}
