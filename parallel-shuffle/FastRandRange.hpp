#pragma once

#include <random>
#include <stdint.h>
#include <limits.h>
#include <assert.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// based on https://stackoverflow.com/questions/51344558/uniform-random-numbers-dont-include-upper-bound
class FastRandRange {
private:
    unsigned int lower_bound;
    unsigned int upper_bound;
    unsigned int threshold;
public:
    // [_lower_bound, _upper_bound)
    FastRandRange(unsigned int _lower_bound, unsigned int _upper_bound) {
        lower_bound = _lower_bound;
        upper_bound = _upper_bound;
        threshold = UINT_MAX - UINT_MAX % (upper_bound - lower_bound);
    }

    template<typename RngT>
    unsigned int rand_in_range(RngT *rng) {
        unsigned int result;
        do {
            result = (*rng)();
        } while (result >= threshold);
        return result % (upper_bound - lower_bound) + lower_bound;
    }
};

// Returns leading zeros as long as x is > 0.
int count_leading_zeros(uint32_t x) {
#ifdef __GNUC__
    return __builtin_clz(x);
#elif defined(_MSC_VER)
    return __lzcnt(x);
#else
    #error

    // This will probably work but be slow.
    // int intlog2 = 0;
    // while (x > 0) {
    //     x >>= 1;
    //     ++intlog2;
    // }
    // return 32 - intlog2;
#endif
}

// Based off of description here http://www.pcg-random.org/posts/bounded-rands.html.
class BitMaskRange {
private:
    uint32_t lower_bound;
    uint32_t range_size;
    uint32_t pow2_mask;
public:
    BitMaskRange(uint32_t _lower_bound, uint32_t _upper_bound) {
        lower_bound = _lower_bound;
        range_size = _upper_bound - _lower_bound;
        pow2_mask = UINT32_MAX >> count_leading_zeros(range_size);
    }

    template<typename RngT>
    unsigned int rand_in_range(RngT *rng) {
        unsigned int result;
        do {
            result = (*rng)() & pow2_mask;
        } while (result >= range_size);
        return result + lower_bound;
    }
};

class LemireRandRange {
private:
    uint32_t lower_bound;
    uint32_t range_size;
public:
    LemireRandRange(uint32_t _lower_bound, uint32_t _upper_bound) {
        lower_bound = _lower_bound;
        range_size = _upper_bound - _lower_bound - 1; // Subtract 1 so we don't need to worry about translating < into <= etc.
    }

    template<typename RngT>
    uint32_t rand_in_range(RngT *rng) {
        uint32_t generated = (*rng)();
        uint64_t widened = uint64_t(generated) * uint64_t(range_size);
        uint32_t narrowed = uint32_t(widened);
        if (narrowed >= range_size) {
            return widened >> 32;
        }

        uint32_t threshold = (-range_size) % range_size;
        do {
            generated = (*rng)();
            widened = uint64_t(generated) * uint64_t(range_size);
            narrowed = uint32_t(widened);
        } while (narrowed < threshold);
        uint32_t result = widened >> 32;
        return result + lower_bound;
    }
};

class LemireRandRangePrecompute {
private:
    uint32_t lower_bound;
    uint32_t range_size;
    uint32_t threshold;
public:
    LemireRandRangePrecompute(uint32_t _lower_bound, uint32_t _upper_bound) {
        lower_bound = _lower_bound;
        range_size = _upper_bound - _lower_bound - 1;
        threshold = (-range_size) % range_size;
    }

    template<typename RngT>
    uint32_t rand_in_range(RngT *rng) {
        uint64_t widened;
        uint32_t narrowed;
        do {
            uint32_t generated = (*rng)();
            widened = uint64_t(generated) * uint64_t(range_size);
            narrowed = uint32_t(widened);
        } while (narrowed < threshold);

        uint32_t result = widened >> 32;
        return result + lower_bound;
    }
};
