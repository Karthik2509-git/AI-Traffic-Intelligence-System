#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <mutex>

namespace antigravity {
namespace core {

/**
 * @brief Managed Pinned Memory Buffer using cudaHostAlloc.
 * 
 * Provides "locked" memory that enables DMA (Direct Memory Access) 
 * transfers, critical for 4K real-time throughput.
 */
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer(size_t count) : count(count) {
        // cudaHostAllocPortable: Memory is visible to all CUDA contexts
        // cudaHostAllocMapped: Maps the memory into the CUDA address space
        cudaError_t err = cudaHostAlloc((void**)&ptr, count * sizeof(T), cudaHostAllocPortable | cudaHostAllocMapped);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned memory: " + std::string(cudaGetErrorString(err)));
        }

        // Retrieve the Device Pointer for use in kernels
        err = cudaHostGetDevicePointer((void**)&d_ptr, (void*)ptr, 0);
        if (err != cudaSuccess) {
            cudaFreeHost(ptr);
            throw std::runtime_error("Failed to get device pointer: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~PinnedBuffer() {
        if (ptr) cudaFreeHost(ptr);
    }

    // Disable copying to prevent double-frees
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* get() { return ptr; }
    const T* get() const { return ptr; }
    
    T* getDevicePtr() { return d_ptr; }
    const T* getDevicePtr() const { return d_ptr; }

    size_t size() const { return count; }
    size_t byte_size() const { return count * sizeof(T); }

private:
    T* ptr = nullptr;
    T* d_ptr = nullptr;
    size_t count;
};

/**
 * @brief Thread-safe Pool for Pinned Memory Buffers.
 * Reduces allocation overhead during high-frequency camera capture.
 */
class MemoryPool {
public:
    static MemoryPool& getInstance() {
        static MemoryPool instance;
        return instance;
    }

    // Allocation/Deallocation logic for specific buffer sizes
    // (To be implemented as needed for the capture pipeline)

private:
    MemoryPool() = default;
};

} // namespace core
} // namespace antigravity
