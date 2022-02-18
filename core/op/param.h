#pragma once

#include "core/base/include/layout.h"

namespace nncore {
namespace param {

struct Reshape {
  Shape new_shape;
};

struct Transpose {
  size_t dimA;
  size_t dimB;
};

struct MatMul {};

struct Dot {};

}  // namespace param

}  // namespace nncore
