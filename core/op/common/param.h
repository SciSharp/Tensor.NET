#pragma once

#include <memory.h>

#include "core/base/include/layout.h"

namespace nncore {
namespace param {

struct reshape {
  Shape new_shape;
};

struct transpose {
  nn_size dimA;
  nn_size dimB;

  transpose(nn_size a, nn_size b) : dimA(a), dimB(b) {}
};

struct permute {
  nn_size* dims;

  permute(const std::vector<nn_size>& value) {
    memcpy(dims, value.data(), value.size() * sizeof(nn_size));
  }
};

struct matmul {};

struct dot {};
}  // namespace param

}  // namespace nncore
