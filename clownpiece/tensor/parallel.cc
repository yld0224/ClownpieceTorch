#include "parallel.h"
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#if CP_PARALLEL_BACKEND != CP_BACKEND_SERIAL && \
    CP_PARALLEL_BACKEND != CP_BACKEND_STD_THREAD && \
    CP_PARALLEL_BACKEND != CP_BACKEND_OPENMP && \
    CP_PARALLEL_BACKEND != CP_BACKEND_THREAD_POOL
#error "The selected parallel backend has not been implemented yet"
#endif

namespace at {
namespace {

#if CP_PARALLEL_BACKEND != CP_BACKEND_SERIAL
unsigned int thread_count() {
  const char* configured = std::getenv("CP_NUM_THREADS");
  if (configured != nullptr) {
    char* parse_end = nullptr;
    long value = std::strtol(configured, &parse_end, 10);
    if (parse_end != configured && *parse_end == '\0' && value > 0) {
      return static_cast<unsigned int>(value);
    }
  }
  unsigned int detected = std::thread::hardware_concurrency();
  return detected == 0 ? 1 : detected;
}
#endif

#if CP_PARALLEL_BACKEND == CP_BACKEND_THREAD_POOL
class ThreadPool {
 private:
  std::queue<std::function<void()>> queue_;
  std::mutex mutex_;
  std::vector<std::thread> threads_;
  std::condition_variable cv_;
  bool stop_ = false;

 public:
  ThreadPool() {
    const unsigned int count = thread_count();
    threads_.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
      threads_.emplace_back([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) {
              return;
            }
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        throw std::runtime_error("ThreadPool: cannot enqueue after shutdown");
      }
      queue_.push(std::move(task));
    }
    cv_.notify_one();
  }
};

struct TaskGroup {
  explicit TaskGroup(int count) : remaining_(count) {}

  std::mutex mutex_;
  std::condition_variable cv_;
  int remaining_;
  std::exception_ptr first_error_;
};
#endif

}  // namespace

void parallel_for(int begin, int end, int grain_size, const RangeTask& task) {
  if (end <= begin) {
    return;
  }
  if (grain_size <= 0) {
    throw std::invalid_argument("parallel_for: grain_size must be positive");
  }

#if CP_PARALLEL_BACKEND == CP_BACKEND_SERIAL
  task(begin, end);
#elif CP_PARALLEL_BACKEND == CP_BACKEND_STD_THREAD
  const int work_size = end - begin;
  const int useful_threads = (work_size + grain_size - 1) / grain_size;
  const int workers = std::max(
      1, std::min(useful_threads, static_cast<int>(thread_count())));

  std::vector<std::thread> threads;
  threads.reserve(workers);
  std::exception_ptr first_error;
  std::mutex error_mutex;

  const int base_size = work_size / workers;
  const int remainder = work_size % workers;
  int range_begin = begin;
  for (int worker = 0; worker < workers; ++worker) {
    const int range_size = base_size + (worker < remainder ? 1 : 0);
    const int range_end = range_begin + range_size;
    threads.emplace_back([&, range_begin, range_end]() {
      try {
        task(range_begin, range_end);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
      }
    });
    range_begin = range_end;
  }

  for (auto& thread : threads) {
    thread.join();
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }
#elif CP_PARALLEL_BACKEND == CP_BACKEND_OPENMP
  const int chunk_count = (end - begin + grain_size - 1) / grain_size;
  const int workers = std::max(
      1, std::min(chunk_count, static_cast<int>(thread_count())));
  std::exception_ptr first_error;
  std::mutex error_mutex;

#pragma omp parallel for schedule(static) num_threads(workers)
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int range_begin = begin + chunk * grain_size;
    const int range_end = std::min(range_begin + grain_size, end);
    try {
      task(range_begin, range_end);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (first_error == nullptr) {
        first_error = std::current_exception();
      }
    }
  }

  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }
#elif CP_PARALLEL_BACKEND == CP_BACKEND_THREAD_POOL
  static ThreadPool pool;
  const int task_count = (end - begin + grain_size - 1) / grain_size;
  auto group = std::make_shared<TaskGroup>(task_count);
  auto shared_task = std::make_shared<RangeTask>(task);
  int submitted = 0;
  for (int first = begin; first < end; first += grain_size) {
    const int last = std::min(first + grain_size, end);
    try {
      pool.enqueue([first, last, group, shared_task]() {
        std::exception_ptr error;
        try {
          (*shared_task)(first, last);
        } catch (...) {
          error = std::current_exception();
        }

        {
          std::lock_guard<std::mutex> lock(group->mutex_);
          if (error != nullptr && group->first_error_ == nullptr) {
            group->first_error_ = error;
          }
          --group->remaining_;
        }
        group->cv_.notify_one();
      });
      ++submitted;
    } catch (...) {
      std::lock_guard<std::mutex> lock(group->mutex_);
      if (group->first_error_ == nullptr) {
        group->first_error_ = std::current_exception();
      }
      group->remaining_ -= task_count - submitted;
      group->cv_.notify_one();
      break;
    }
  }
  std::unique_lock<std::mutex> lock(group->mutex_);
  group->cv_.wait(lock, [&group]() { return group->remaining_ == 0; });
  auto error = group->first_error_;
  lock.unlock();
  if (error != nullptr) {
    std::rethrow_exception(error);
  }
#endif
}

}  // namespace at
