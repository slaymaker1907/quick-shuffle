#include <cstdint>

class XorShift128 {
private:
    uint32_t x, y, z, w;
public:
    XorShift128() {
        x = 123456789;
        y = 362436069;
        z = 521288629;
        w = 88675123;
    }
    XorShift128(XorShift128 *other) {
        x = (*other)();
        y = (*other)();
        z = (*other)();
        w = (*other)();
    }
    uint32_t operator()() {
        uint32_t t = x ^ (x << 11);   
        x = y; y = z; z = w;   
        return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
    }
};
