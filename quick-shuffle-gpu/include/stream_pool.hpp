#ifndef STREAM_POOL
#define STREAM_POOL

#include "cuda_polyfill.hpp"
#include "permute_util.hpp"
#include <cstdlib>
#include <vector>

namespace cuda_permute
{

template<typename T>
T* alloc_pinned(T *dummy = nullptr) {
    T *result;
    check_cuda_error(cudaMallocHost((void**)&result, sizeof(T)));
    return result;
}

template<typename T>
T* alloc_pinned_array(size_t tcount, T *dummy = nullptr) {
    T *result;
    check_cuda_error(cudaMallocHost((void**)&result, tcount * sizeof(T)));
    return result;
}

template<typename T>
void memcpy_array(T *dest, T *src, size_t tcount) {
    memcpy(dest, src, sizeof(T) * tcount);
}

void free_pinned(void *to_free) {
    check_cuda_error(cudaFreeHost(to_free));
}

class CudaKernelExample {
public:
    void run(cudaStream_t stream) {
        // launch kernel in here.
    }
};

class CudaEvent {
private:
    cudaEvent_t event;
    bool is_init;
public:
    CudaEvent() {
        is_init = false;
        check_cuda_error(cudaEventCreate(&event));
        is_init = true;
    }

    ~CudaEvent() {
        if (is_init) {
            cudaEventDestroy(event);
            is_init = false;
        }
    }

    void record_event(cudaStream_t stream) {
        check_cuda_error(cudaEventRecord(event, stream));
    }

    void wait(cudaStream_t stream) {
        check_cuda_error(cudaStreamWaitEvent(stream, event, 0));
    }
};

class CudaStream {
private:
    cudaStream_t wrapped_stream;
    bool is_stream_init;
public:
    CudaStream() {
        is_stream_init = false;
        check_cuda_error(cudaStreamCreate(&wrapped_stream));
        is_stream_init = true;
    }
    ~CudaStream() {
        if (is_stream_init) {
            cudaStreamDestroy(wrapped_stream);
            is_stream_init = false;
        }
    }

    void join() {
        check_cuda_error(cudaStreamSynchronize(wrapped_stream));
    }

    void record_event(CudaEvent *event) {
        event->record_event(wrapped_stream);
    }

    CudaEvent* create_event() {
        CudaEvent *event = new CudaEvent();
        record_event(event);
        return event;
    }

    void wait_for_event(CudaEvent *event) {
        event->wait(wrapped_stream);
    }

    template<typename Kernel>
    void launch_kernel(Kernel kernel) {
        kernel.run(wrapped_stream);
    }

    void general_memcpy(void *dest, void *src, size_t size, cudaMemcpyKind direction) {
        check_cuda_error(cudaMemcpyAsync(dest, src, size, direction, wrapped_stream));
    }

    template<typename T>
    void stream_memcpy(T *dest, T *src, cudaMemcpyKind direction, size_t tcount = 1) {
        general_memcpy(dest, src, tcount * sizeof(T), direction);
    }

    template<typename T>
    void memcpy_to_device(T *dest, T *src, size_t tcount = 1) {
        stream_memcpy(dest, src, cudaMemcpyHostToDevice, tcount);
    }

    template<typename T>
    void memcpy_to_host(T *dest, T *src, size_t tcount = 1) {
        stream_memcpy(dest, src, cudaMemcpyDeviceToHost, tcount);
    }

    // must call join before calling free_pinned NOT free.
    // may free or reuse host_ptr after calling this function.
    template<typename T>
    T* memcpy_unpinned(T *device_ptr, T *host_ptr, size_t tcount = 1) {
        T *host_pinned = alloc_pinned_array(tcount, device_ptr);
        memcpy_array(host_pinned, host_ptr, tcount);
        memcpy_to_device(device_ptr, host_pinned, tcount);
        return host_pinned;
    }

    // may be done asynchronously so need to join before using.
    // must be freed with free_pinned.
    template<typename T>
    T* memcpy_to_host(T* device_ptr, size_t tcount = 1) {
        T *host_pinned = alloc_pinned_array(tcount, device_ptr);
        memcpy_to_host(device_ptr, host_pinned, tcount);
        return host_pinned;
    }
};

void wait_for_all(const std::vector<CudaStream*> &streams) {
    for (CudaStream *stream : streams) {
        stream->join();
    }
}

}

#endif
