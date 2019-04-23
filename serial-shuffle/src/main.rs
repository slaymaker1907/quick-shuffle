#![feature(test)]

extern crate rand;
extern crate test;

pub mod lib;

use rand::{Rng};
use rand::FromEntropy;
use rand::rngs::SmallRng;
use std::time::{Instant};
use std::cmp::{min, max};
use std::mem::{size_of, swap, transmute};
use std::ptr;

pub fn fischer_yates<R: Rng+?Sized, T>(rng: &mut R, data: &mut [T]) {
    // Do unsafe to avoid bunch of bounds checks.
    let data_len = data.len();
    unsafe {
        for i in 0..data_len {
            let swap_pos = rng.gen_range(i, data_len);
            let first = data.get_unchecked_mut(i) as *mut T;
            let second = data.get_unchecked_mut(swap_pos) as *mut T;
            ptr::swap(first, second);
        }
    }
}

#[derive(Copy, Clone)]
struct ShuffleHelper<T: Copy> {
    data: T,
    pos: u64
}

pub fn sort_shuffle<R: Rng+?Sized, T: Copy>(rng: &mut R, data: &mut [T]) {
    // First do a basic shuffle, then randomize runs using fischer-yates for correctness.
    // Can't use a dictionary since that is worst case O(sqrt(n))!
    let mut to_sort: Vec<ShuffleHelper<T>> = data.iter().map(|&x| ShuffleHelper{
        data: x,
        pos: rng.next_u64()
    }).collect();
    to_sort.sort_unstable_by_key(|helper| helper.pos);

    let mut i: usize = 0;
    let end_pt = to_sort.len() - 1;
    while i < end_pt {
        unsafe {
            let first_ele = *to_sort.get_unchecked(i);
            let value = first_ele.pos;
            let second_ele = *to_sort.get_unchecked(i+1);

            if value == second_ele.pos {
                let first_ind = i;
                i += 2;

                while i < end_pt {
                    let temp = to_sort.get_unchecked(i).pos;
                    if temp != value {
                        break;
                    }
                }

                // Now shuffle from first_ind to i, end exclusive.
                fischer_yates(rng, &mut to_sort[first_ind..i]);
            } else {
                i += 1;
            }
        }
    }

    for i in 0..data.len() {
        unsafe {
            *data.get_unchecked_mut(i) = to_sort.get_unchecked(i).data;
        }
    }
}

// Close enought to correct for my purposes.
fn isqrt(x: usize) -> usize {
    (x as f64).sqrt().ceil() as usize
}

struct HashArrayTree<T> {
    rows: Vec<Vec<T>>,
    row_length: usize
}

impl<T> HashArrayTree<T> {
    fn new(init_size: usize) -> HashArrayTree<T> {
        let mut rows: Vec<Vec<T>> = Vec::with_capacity(init_size);
        rows.push(Vec::with_capacity(init_size));
        return HashArrayTree{
            rows,
            row_length: init_size
        };
    }

    // fn iter_mut(&mut self) -> impl Iterator<Item=&mut T> {
    //     self.rows.iter_mut().flat_map(|row| row.iter_mut())
    // }

    fn get_index(&self, ind: usize) -> (usize, usize) {
        let bit_shift = self.row_length.trailing_zeros();
        let row_ind = ind >> bit_shift;
        let col_ind = ind - (row_ind << bit_shift);
        (row_ind, col_ind)
    }

    unsafe fn unchecked_swap(&mut self, ind1: usize, ind2: usize) {
        let (row1, col1) = self.get_index(ind1);
        let (row2, col2) = self.get_index(ind2);

        let ptr1 = {
            let row = self.rows.get_unchecked_mut(row1);
            row.get_unchecked_mut(col1) as *mut T
        };
        let ptr2 = {
            let row = self.rows.get_unchecked_mut(row2);
            row.get_unchecked_mut(col2) as *mut T
        };

        ptr::swap(ptr1, ptr2);
    }

    fn len(&self) -> usize {
        let row_len = self.rows.len();
        if row_len == 0 {
            0
        } else {
            unsafe {
                self.rows.get_unchecked(row_len - 1).len() + self.row_length * (row_len - 1)
            }
        }
    }

    fn push(&mut self, item: T) {
        let row_len = self.row_length;
        let last_ind = self.rows.len() - 1;
        unsafe {
            let last_len = self.rows.get_unchecked(last_ind).len();
            let last_row = if last_len == row_len {
                if last_ind + 1 == row_len {
                    let new_size = row_len * 2;
                    let mut new_rows: Vec<Vec<T>> = Vec::with_capacity(new_size);
                    new_rows.push(Vec::with_capacity(new_size));
                    swap(&mut self.rows, &mut new_rows);
                    let mut current_row = self.rows.get_unchecked_mut(0) as *mut Vec<T>;
                    let old_rows = new_rows; // Just changing name to prevent confusion.

                    for row in old_rows {
                        for ele in row {
                            let my_ref = &mut *current_row;
                            my_ref.push(ele);
                            if my_ref.len() == new_size {
                                self.rows.push(Vec::with_capacity(new_size));
                                let last_ind = self.rows.len() - 1;
                                current_row = self.rows.get_unchecked_mut(last_ind) as *mut Vec<T>;
                            }
                        }
                    }

                    self.row_length = new_size;
                    let new_last_ind = self.rows.len();
                    self.rows.get_unchecked_mut(new_last_ind - 1)
                } else {
                    self.rows.push(Vec::with_capacity(row_len));
                    self.rows.get_unchecked_mut(last_ind + 1)
                }
            } else {
                self.rows.get_unchecked_mut(last_ind)
            };

            last_row.push(item);
        }
    }

    // fn transfer(self, dest: &mut [T]) {
    //     let mut iter_mut = dest.iter_mut();
    //     for row in self.rows {
    //         for ele in row {
    //             *iter_mut.next().unwrap() = ele;
    //         }
    //     }
    // }

    fn into_iter(self) -> impl Iterator<Item=T> {
        self.rows.into_iter().flatten()
    }

    fn shuffle<R: Rng+?Sized>(&mut self, rng: &mut R) {
        let leng = self.len();
        for i1 in 0..leng {
            let i2 = rng.gen_range(i1, leng);
            if i1 != i2 {
                unsafe {
                    self.unchecked_swap(i1, i2);
                }
            }
        }
    }
}

const fn eles_per_page<T>() -> usize {
    // let tsize = size_of::<T>();
    // let page_size = 64 * 1024; // Taken for my PC's L1 cache.
    // page_size / tsize
    (64 * 1024) / size_of::<T>()
}

// const fn fy_cutoff<T>() -> usize {
//     512 * 1024 * 1024 / size_of::<T>()
// }

const fn max_partitions<T>() -> usize {
    // let cache_line_size = 64;
    // let page_size = 64 * 1024;
    // let hat_end_size = size_of::<HashArrayTree<T>>() + 2 * cache_line_size;
    (64 * 1024) / (size_of::<HashArrayTree<T>>() + 2 * 64)
}

pub fn quick_shuffle<'a, R: Rng+?Sized, T: Copy>(rng: &'a mut R, data: impl Iterator<Item=T>) -> Vec<T> {
    let total_size = data.size_hint().1.unwrap_or(usize::max_value());
    let eles_ppage = max(eles_per_page::<T>(), 1);
    let ideal_size = max(total_size / eles_ppage, 2);
    let pcount = min(ideal_size, max(max_partitions::<T>(), 2));
    let exp_size = total_size / pcount + 1;
    let mut partitions: Vec<HashArrayTree<T>> = Vec::with_capacity(pcount);
    let init_hat_size = isqrt(exp_size).next_power_of_two();
    for _ in 0..pcount {
        partitions.push(HashArrayTree::new(init_hat_size));
    }

    for datum in data {
        let pnum = rng.gen_range(0, pcount);
        unsafe {
            partitions.get_unchecked_mut(pnum).push(datum);
        }
    }

    partitions.into_iter().flat_map(|mut partition| {
        // TODO: Fix type system so that we can go back recursively.
        partition.shuffle(rng);
        partition.into_iter()
    }).collect()
}

// Maybe try to partition by sorting?
pub fn quick_shuffle_big<R: Rng+?Sized, T: Copy>(rng: &mut R, data: &mut [T]) {
    let total_size = data.len();
    let tsize = size_of::<T>();
    let page_size = 64 * 1024; // Taken for my PC's L1 cache.
    let eles_per_page = page_size / tsize;
    if total_size <= eles_per_page {
        fischer_yates(rng, data);
        return;
    }

    let ideal_size = max(total_size / eles_per_page, 2);
    let pcount = ideal_size;
    let mut partitioned: Vec<(T, usize)> = data.iter().map(|&ele| (ele, rng.gen_range(0, pcount))).collect();
    partitioned.sort_unstable_by_key(|&(_, part)| part);

    let mut partitioned_it = partitioned.into_iter();
    let mut part_start = 0usize;
    let first_ele = partitioned_it.next().unwrap();
    let mut current_part = first_ele.1;
    unsafe {
        *data.get_unchecked_mut(0) = first_ele.0;
    }
    let mut pos = 1usize;

    for (ele, part) in partitioned_it {
        unsafe {
            *data.get_unchecked_mut(pos) = ele;
        }
        pos += 1;

        if part != current_part {
            let to_shuffle = &mut data[part_start..pos];
            fischer_yates(rng, to_shuffle); // Just assume it is pretty small.
            part_start = pos;
            current_part = part;
        }
    }

    let last_part = &mut data[part_start..total_size];
    fischer_yates(rng, last_part);
}

pub fn merge_shuffle<R: Rng+?Sized, T>(rng: &mut R, data: &mut [T]) {
    let fy_cutoff = 16 * 1024 * 1024;
    let total_len = data.len();
    if total_len * size_of::<T>() <= fy_cutoff {
        fischer_yates(rng, data);
        return;
    }

    let half_len = total_len / 2;
    let mut left_swap = {
        let (left, right) = data.split_at_mut(half_len);

        merge_shuffle(rng, left);
        merge_shuffle(rng, right);
        let mut left_copy: Vec<T> = Vec::with_capacity(half_len);
        unsafe {
            left_copy.set_len(half_len); // It is now filled with a bunch of garbage. Need to be careful to make sure never dropped.
        }

        for pos in 0..half_len {
            unsafe {
                ptr::swap(left_copy.get_unchecked_mut(pos) as *mut T, left.get_unchecked_mut(pos) as *mut T);
            }
        }
        left_copy
    };

    let mut left_pos = 0usize;
    let mut right_pos = total_len - half_len;
    let mut ins_pos = 0usize;

    while left_pos < half_len && right_pos < total_len {
        let left_rem = half_len - left_pos;
        let right_rem = total_len - right_pos;
        let rand_num = rng.gen_range(0, left_rem + right_rem);
        if rand_num < left_rem {
            unsafe {
                swap(left_swap.get_unchecked_mut(left_pos), data.get_unchecked_mut(ins_pos));
            }
            left_pos += 1;
        } else {
            unsafe {
                unsafe_slice_swap(data, right_pos, ins_pos);
            }
            right_pos += 1;
        }

        ins_pos += 1;
    }

    while left_pos < half_len {
        unsafe {
            swap(left_swap.get_unchecked_mut(left_pos), data.get_unchecked_mut(ins_pos));
        }
        left_pos += 1;
        ins_pos += 1;
    }
    // If we get to here, right is already in position.
    // Make sure to set len again so it is not dropped.
    unsafe {
        left_swap.set_len(0);
    }
}

unsafe fn unsafe_slice_swap<T>(data: &mut [T], a: usize, b: usize) {
    let ptr1 = data.get_unchecked_mut(a) as *mut T;
    let ptr2 = data.get_unchecked_mut(b) as *mut T;
    ptr::swap(ptr1, ptr2);
}

pub fn inplace_quick_shuffle<R: Rng+?Sized, T>(rng: &mut R, mut data: &mut [T]) {
    // Do a loop to use tail recursion and avoid O(n) worst case space.
    loop {
        let total_len = data.len();
        let fy_cutoff = 512 * 1024 * 1024;
        if total_len * size_of::<T>() <= fy_cutoff {
            fischer_yates(rng, data);
            return;
        }

        let mut left_pos = 0usize;
        let mut right_pos = total_len - 1;
        let mut left_part: bool = rng.gen(); // True indicates left.
        let mut right_part: bool = rng.gen();

        while left_pos < right_pos {
            if left_part && right_part {
                left_pos += 1;
                left_part = rng.gen();
            } else if !left_part && right_part {
                unsafe {
                    unsafe_slice_swap(data, left_pos, right_pos);
                    // data.swap(left_pos, right_pos);
                    left_pos += 1;
                    right_pos -= 1;
                }
                left_part = rng.gen();
                right_part = rng.gen();
            } else if left_part && !right_part {
                left_pos += 1;
                right_pos -= 1;
                left_part = rng.gen();
                right_part = rng.gen();
            } else {
                right_pos -= 1;
                right_part = rng.gen();
            }
        }

        // left is 0..part_ind, right is part_ind..total_len.
        let part_ind = if left_pos == right_pos {
            if left_part {
                left_pos + 1
            } else {
                left_pos
            }
        } else {
            left_pos
        };

        let (left, right) = data.split_at_mut(part_ind);
        if left.len() >= right.len() {
            inplace_quick_shuffle(rng, left);
            data = right;
        } else {
            inplace_quick_shuffle(rng, right);
            data = left;
        }
    }
}

fn standard_vec<R: Rng+?Sized>(rng: &mut R, size: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(size);
    for _ in 0..size {
        result.push(rng.gen());
    }
    result
}

fn group_partitions<T1, T2, K, F>(mut data: impl Iterator<Item=T1>, mut func: F) -> HashArrayTree<HashArrayTree<T2>>
    where F: FnMut(T1) -> (T2, K),
          K: Eq
{
    let mut result: HashArrayTree<HashArrayTree<T2>> = HashArrayTree::new(4);
    if let Some(first_datum) = data.next() {
        let (datum, mut current_key) = func(first_datum);
        let mut current_table: HashArrayTree<T2> = HashArrayTree::new(4);
        current_table.push(datum);

        for datum in data {
            let (datum, next_key) = func(datum);
            if current_key != next_key {
                result.push(current_table);
                current_key = next_key;
                current_table = HashArrayTree::new(4);
            }
            current_table.push(datum);
        }

        result.push(current_table);
    }

    result
}

pub fn limited_partition_sort_shuffle<T, R: Rng+?Sized>(rng: &mut R, data: impl Iterator<Item=T>) -> Vec<T> {
    let size_est = data.size_hint().1.unwrap_or(usize::max_value());
    let pcount = isqrt(size_est);
    let partitions = partition_by_sort(data, |datum| {
        (datum, rng.gen_range(0, pcount))
    });

    partitions.into_iter().flat_map(|mut part| {
        if part.len() <= 2 {
            part.shuffle(rng);
            let temp_vec: Vec<T> = part.into_iter().collect();
            temp_vec.into_iter()
        } else {
            limited_partition_sort_shuffle(rng, part.into_iter()).into_iter()
        }
    }).collect()
}

fn partition_by_sort<T, K: Ord, F>(data: impl Iterator<Item=T>, mut func: F) -> HashArrayTree<HashArrayTree<T>>
    where F: FnMut(T) -> (T, K) 
{
    let mut mapped_data: Vec<(T, K)> = data.map(|datum| func(datum)).collect();
    sort_with_dups(&mut mapped_data);
    group_partitions(mapped_data.into_iter(), |(data, key)| {
        (data, key)
    })
}

fn partition_by_hash<T, F>(data: impl Iterator<Item=T>, mut func: F, size: usize) -> Vec<HashArrayTree<T>>
    where F: FnMut(T) -> (T, usize)
{
    let mut result: Vec<HashArrayTree<T>> = Vec::with_capacity(size);
    for _ in 0..size {
        result.push(HashArrayTree::new(4));
    }

    for datum in data {
        unsafe {
            let (datum, part) = func(datum);
            result.get_unchecked_mut(part).push(datum);
        }
    }

    result
}

fn sort_with_dups<T, K: Ord>(data: &mut [(T, K)]) {
    let mut rng = SmallRng::from_entropy();
    sort_with_dups_helper(&mut rng, data);
}

unsafe fn get_key_unchecked<T, K>(data: &[(T, K)], ind: usize) -> &K {
    &data.get_unchecked(ind).1
}

fn sort_with_dups_helper<T, K: Ord, R: Rng+?Sized>(rng: &mut R, data: &mut [(T, K)]) {
    let total_len = data.len();
    if total_len <= 1 {
        return;
    } else if total_len == 2 {
        let cmp_result = unsafe {
            let first_key = get_key_unchecked(data, 0);
            let second_key = get_key_unchecked(data, 1);
            first_key > second_key
        };

        if cmp_result {
            unsafe {
                unsafe_slice_swap(data, 0, 1);
            }
        }

        return;
    }

    let mut ind1 = 0usize.wrapping_sub(1usize);
    let mut ind2 = total_len;
    // let mut pivot_ind = median_of_medians(data, |datum| &datum.1); // Has to be mutable so we don't need to copy.
    let mut pivot_ind = rng.gen_range(0, total_len);
    let mut left_distinct = false;
    let mut right_distinct = false;

    // Check for distinction for correctly placed elements and when we swap.
    let last_right = total_len - 1;

    loop {
        unsafe {
            let pivot_ref = get_key_unchecked(data, pivot_ind);
            loop {
                ind1 = ind1.wrapping_add(1usize);
                let current = get_key_unchecked(data, ind1);
                if current >= pivot_ref {
                    break;
                }

                if !left_distinct {
                    left_distinct = ind1 > 0 && current != get_key_unchecked(data, ind1 - 1);
                }
            }

            while ind2 > 0 {
                // if ind2 == 0 {
                //     println!("Here");
                //     println!("{}, {}, {}", ind1, ind2, total_len);
                // }
                ind2 -= 1;
                let current = get_key_unchecked(data, ind2);
                if current <= pivot_ref {
                    break;
                }

                if !right_distinct {
                    right_distinct = ind2 < last_right && current != get_key_unchecked(data, ind1 + 1);
                }
            }
        }

        if ind1 >= ind2 {
            break;
        }
        unsafe {
            unsafe_slice_swap(data, ind1, ind2);
        }

        // Check to see if we swapped the pivot and adjust index accordingly.
        if ind1 == pivot_ind {
            pivot_ind = ind2;
        } else if ind2 == pivot_ind {
            pivot_ind = ind1;
        }

        // Do a final check for distinctness since we just swapped something in that might be different.
        // Hopefully branch predictor makes this efficient.
        unsafe {
            if !left_distinct {
                left_distinct = ind1 > 0 && get_key_unchecked(data, ind1) != get_key_unchecked(data, ind1 - 1);
            }
            if !right_distinct {
                right_distinct = ind2 < last_right && get_key_unchecked(data, ind2) != get_key_unchecked(data, ind1 + 1);
            }
        }
    }

    let (left, right) = data.split_at_mut(ind2 + 1);
    unsafe {
        if left.len() == 0 && right_distinct {
            let data_copy = transmute::<&mut [(T, K)], &mut [(u64, usize)]>(right);
            let my_vec: Vec<usize> = data_copy.iter().map(|&tup| tup.1).collect();
            println!("{:?}", my_vec);
            println!("{}, {}", ind1, ind2);
            panic!("Should never get here");
        }
    }
    if left_distinct {
        sort_with_dups_helper(rng, left);
    }
    if right_distinct {
        sort_with_dups_helper(rng, right);
    }
}

fn median_of_medians<T, K, F>(data: &[T], func: F) -> usize
    where K: Ord,
          F: Fn(&T) -> &K
{
    let mut medians: Vec<usize> = data.chunks(5).map(|chunk| {
        unsafe {
            median_of_five(chunk, |datum| func(datum))
        }
    }).collect();

    while medians.len() > 5 {
        medians = medians.chunks(5).map(|chunk| {
            unsafe {
                median_of_five(chunk, |&index| {
                    // Have to cast this to a static lifetime since rustc can't verify the correctness.
                    let datum = &*(data.get_unchecked(index) as *const T);
                    &*(func(datum) as *const K)
                })
            }
        }).collect()
    }

    unsafe {
        median_of_five(&medians, |&index| {
            let datum = &*(data.get_unchecked(index) as *const T);
            &*(func(datum) as *const K)
        })
    }
}

// fn median_of_medians_helper<T: Ord>(data: &[T], indices: 

unsafe fn median_of_five<T, K, F>(data: &[T], func: F) -> usize
    where K: Ord,
          F: Fn(&T) -> &K
{
    let total_len = data.len();
    let mut inds_arr: [usize; 5] = [0, 1, 2, 3, 4];

    for i1 in 1..total_len {
        let mut i2 = i1.wrapping_sub(1);
        while i2 < 5  { // We will have wrap around.
            let need_swap = {
                let true_ind = *inds_arr.get_unchecked(i2 + 1);
                let inserting = func(data.get_unchecked(true_ind));
                let true_ind2 = *inds_arr.get_unchecked(i2);
                inserting < func(data.get_unchecked(true_ind2))
            };

            if need_swap {
                unsafe_slice_swap(&mut inds_arr, i2, i2 + 1);
                i2 = i2.wrapping_sub(1);
            } else {
                break;
            }
        }
    }

    inds_arr[total_len / 2]
}

static SHUFF_SIZE: usize = 100_000_000;

fn main() {
    let mut gen = SmallRng::from_entropy();
    let mut to_shuffle = standard_vec(&mut gen, SHUFF_SIZE); // Make this thing truly enormous.

    let start = Instant::now();
    fischer_yates(&mut gen, &mut to_shuffle);
    let end_time = start.elapsed();
    println!("Fischer-Yates took: {}.{}s", end_time.as_secs(), end_time.subsec_millis());

    let start = Instant::now();
    sort_shuffle(&mut gen, &mut to_shuffle);
    let end_time = start.elapsed();
    println!("Sort-Shuffle took: {}.{}s", end_time.as_secs(), end_time.subsec_millis());

    let quick_start = Instant::now();
    to_shuffle = quick_shuffle(&mut gen, to_shuffle.into_iter());
    let quick_time = quick_start.elapsed();
    println!("Quick-Shuffle (Hash) took: {}.{}s", quick_time.as_secs(), quick_time.subsec_millis());

    // let quick_start = Instant::now();
    // to_shuffle = limited_partition_sort_shuffle(&mut gen, to_shuffle.into_iter());
    // let quick_time = quick_start.elapsed();
    // println!("Quick-Shuffle (Full Sort) took: {}.{}s", quick_time.as_secs(), quick_time.subsec_millis());

    let start = Instant::now();
    inplace_quick_shuffle(&mut gen, &mut to_shuffle);
    let end_time = start.elapsed();
    println!("Quick-Shuffle (in-place) took: {}.{}s", end_time.as_secs(), end_time.subsec_millis());

    let partitions = 50_000;

    let quick_start = Instant::now();
    to_shuffle = partition_by_sort(to_shuffle.into_iter(), |datum| (datum, gen.gen_range(0, partitions)))
        .into_iter()
        .flat_map(|part| part.into_iter())
        .collect();
    let quick_time = quick_start.elapsed();
    println!("Sort Partition Took: {}.{}s", quick_time.as_secs(), quick_time.subsec_millis());

    let quick_start = Instant::now();
    to_shuffle = partition_by_hash(to_shuffle.into_iter(), |datum| (datum, gen.gen_range(0, partitions)), partitions)
        .into_iter()
        .flat_map(|part| part.into_iter())
        .collect();
    let quick_time = quick_start.elapsed();
    println!("Hash Partition Took: {}.{}s", quick_time.as_secs(), quick_time.subsec_millis());

    test::black_box(to_shuffle);
}
