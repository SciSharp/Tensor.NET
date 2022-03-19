#pragma once

#include <memory.h>

#include "core/base/include/layout.h"

namespace nncore {
namespace param {

struct reshape {
  Shape new_shape;
};

struct transpose {
  size_t dimA;
  size_t dimB;

  transpose(size_t a, size_t b) : dimA(a), dimB(b) {}
};

struct permute {
  size_t dims[NN_MAX_NDIM];

  permute(const std::vector<size_t>& value) {
    memcpy(dims, value.data(), value.size() * sizeof(size_t));
  }
};

struct matmul {};

struct dot {};

}  // namespace param

}  // namespace nncore
