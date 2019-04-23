use std::alloc::{alloc, dealloc, Layout}; // Maybe change to use a specified allocator in future.
use std::ptr::{null_mut};
use std::ops::{Add, Mul};
use std::mem::size_of;
use std::marker::PhantomData;
use std::cmp::max;

pub trait PointerSize: Add+Sized+Mul {
    fn zero() -> Self;
}

impl PointerSize for u32 {
    fn zero() -> Self {
        0
    }
}

impl PointerSize for u64 {
    fn zero() -> Self {
        0
    }
}

// Probably doesn't make sense to go smaller than this.
impl PointerSize for u16 {
    fn zero() -> Self {
        0
    }
}

struct BasicArray<T, SizeT: PointerSize> {
    size: SizeT,
    data: *mut T
}

pub struct MemoryRegion<SizeT: PointerSize> {
    top_level: Vec<RegionNode<SizeT>>,
    block_size: usize
}

pub enum RegionReference<'a, T, SizeT: PointerSize> {
    RawReference(*mut T),
    ComplexReference{
        // Have a reference here instead of pointer to give better lifetime guarantees.
        region: *mut MemoryRegion<SizeT>,
        address: SizeT,
        phantom: PhantomData<&'a MemoryRegion<SizeT>>
    }
}

struct RegionNode<SizeT: PointerSize> {
    data: BasicArray<u8, SizeT>, // size given here may be an underestimate if node was freed.
    left: *mut RegionNode<SizeT>,
    right: *mut RegionNode<SizeT>,
    parent: *mut RegionNode<SizeT>, // Maybe take out parent pointer later.
    total_size: usize,
    depth: u8
}

impl<SizeT: PointerSize> MemoryRegion<SizeT> {
    pub fn new(block_size: usize) -> MemoryRegion<SizeT> {
        MemoryRegion {
            top_level: Vec::new(),
            block_size: block_size + size_of::<RegionNode<SizeT>>()
        }
    }

    pub fn alloc<'a, T>(&'a mut self, data: T) -> RegionReference<'a, T, SizeT> {
        let obj_size = size_of::<T>();
        let main_node = if self.top_level.len() == 0 {
            unsafe {
                let new_size = max(obj_size, self.block_size);
                let layout = Layout::from_size_align_unchecked(new_size, 8); // Be conservative with alignment.
                let raw_mem = alloc(layout);
                let region_ptr = raw_mem as *mut RegionNode<SizeT>;
                // let array = 
            }
        } else {
        };

        unimplemented!()
    }
}
