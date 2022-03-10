#pragma once

#include "core/base/include/layout.h"

namespace nncore {
namespace param {

struct reshape {
  Shape new_shape;
};

struct transpose {
  size_t dimA;
  size_t dimB;
};

struct matmul {};

struct dot {};

}  // namespace param

}  // namespace nncore
