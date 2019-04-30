#pragma once

#include <random>
#include <stdint.h>
#include <limits.h>

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
