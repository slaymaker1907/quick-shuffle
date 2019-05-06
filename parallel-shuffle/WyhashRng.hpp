#include <cstdint>

// Taken from https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/
// only works with GCC due to uint128.
class WyhashRng {
private:
    uint64_t state;
public:
    WyhashRng(uint64_t seed) {
        state = seed;
    }
    uint32_t operator()() {
        state += 0x60bee2bee120fc15;
        __uint128_t tmp;
        tmp = (__uint128_t)state * 0xa3b195354a39b70d;
        uint64_t m1 = (tmp >> 64) ^ tmp;
        tmp = (__uint128_t)m1 * 0x1b03738712fad5c9;
        uint64_t m2 = (tmp >> 64) ^ tmp;
        return m2;
    }
};