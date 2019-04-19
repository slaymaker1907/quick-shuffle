#ifndef PARTIAL_HAT
#define PARTIAL_HAT

// TODO: finish this thing.

#include <cmath>
#include <assert.h>

namespace cuda_permute
{
size_t div_ceil(size_t to_divide, size_t divisor) {
    size_t result = to_divide / divisor;
    if (result * divisor < divisor) {
        ++result;
    }
    return result;
}

// This class is an array of arrays which is initially statically allocated.
// The advantage is that it can be deallocated as you iterate through it with a max overhead of sqrt(size).
// write for float, then change to T.
// template <class T>
class PartialHAT {
private:
    int **data_columns;
    size_t column_size;
    size_t total_size;
public:
    // size should be > 0.
    PartialHAT(size_t total_input_size) {
        total_size = total_input_size;
        double dbsqrt = sqrt((double)total_size); // probably good enough and faster on a GPU.
        column_size = (size_t)ceil(dbsqrt);
        size_t column_byte_size = column_size * sizeof(int);
        size_t needed_columns = div_ceil(total_size, column_size);

        // first allocate array of arrays.
        data_columns = (int**)malloc(needed_columns * sizeof(int*));
        assert(data_columns);

        // now allocate each column.
        for (int i = 0; i < needed_columns; i++) {
            data_columns[i] = (int*)malloc(column_byte_size);
            assert(data_columns[i]);
        }
    }
};

}
#endif