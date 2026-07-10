#ifndef PARALLEL_H
#define PARALLEL_H 1

#include <functional>

#define CP_BACKEND_SERIAL 0
#define CP_BACKEND_STD_THREAD 1
#define CP_BACKEND_OPENMP 2
#define CP_BACKEND_THREAD_POOL 3

#ifndef CP_PARALLEL_BACKEND
#define CP_PARALLEL_BACKEND CP_BACKEND_SERIAL
#endif

namespace at {

using RangeTask = std::function<void(int, int)>;

void parallel_for(int begin, int end, int grain_size, const RangeTask& task);

}  // namespace at

#endif
