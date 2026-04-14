#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace antigravity {
namespace core {

/**
 * @brief High-throughput, thread-safe queue for the processing pipeline.
 * 
 * Used to pass frame buffers and analytical results between asynchronous 
 * stages (Capture -> Inference -> Analytics).
 */
template <typename T>
class ConcurrentQueue {
public:
    ConcurrentQueue(size_t maxSize = 0) : max_size(maxSize) {}

    void push(T&& value) {
        std::unique_lock<std::mutex> lock(mtx);
        
        // If queue is full, wait for space (optional based on max_size)
        if (max_size > 0) {
            cond_full.wait(lock, [this]() { return queue.size() < max_size || stop_flag; });
        }

        if (stop_flag) return;

        queue.push(std::move(value));
        lock.unlock();
        cond_empty.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx);
        
        cond_empty.wait(lock, [this]() { return !queue.empty() || stop_flag; });

        if (stop_flag && queue.empty()) return false;

        value = std::move(queue.front());
        queue.pop();
        
        lock.unlock();
        cond_full.notify_one();
        
        return true;
    }

    void stop() {
        stop_flag = true;
        cond_empty.notify_all();
        cond_full.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

private:
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cond_empty;
    std::condition_variable cond_full;
    size_t max_size;
    std::atomic<bool> stop_flag{false};
};

} // namespace core
} // namespace antigravity
