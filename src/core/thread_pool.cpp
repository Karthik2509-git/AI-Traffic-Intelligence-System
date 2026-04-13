#include "core/thread_pool.hpp"

namespace atos {
namespace core {

ThreadPool::ThreadPool(size_t threads) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { 
                        return this->stop_flag || !this->tasks.empty(); 
                    });

                    if (this->stop_flag && this->tasks.empty()) return;

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_flag = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

} // namespace core
} // namespace atos
