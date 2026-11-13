//! `download_into` and `upload_into` allocate nothing on the heap. Counted per thread, so
//! parallel tests do not interfere.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use linalg_core::{Backend, DType};
use linalg_cpu::CpuBackend;

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
fn download_into_and_upload_into_do_not_allocate() {
    let be = CpuBackend::new();
    for dtype in [DType::F32, DType::BF16] {
        let mut t = be.alloc(dtype, &[256, 64]).unwrap();
        let src: Vec<u8> = (0..256 * 64 * dtype.size_of()).map(|i| (i % 251) as u8).collect();
        let mut out = vec![0u8; src.len()];
        let (n, r) = allocs_during(|| be.upload_into(&src, &mut t));
        assert_eq!((n, r), (0, Ok(())), "upload_into {dtype:?}");
        let (n, r) = allocs_during(|| be.download_into(&t, &mut out));
        assert_eq!((n, r), (0, Ok(())), "download_into {dtype:?}");
        assert_eq!(out, src);
    }
}
