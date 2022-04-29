#include <cmath>
#include <iostream>
#include <random>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SELF_MODIFY_INTERNAL(fill) {
  nn_size n = layout.total_elems();
  for (nn_size i = 0; i < n; i++) {
    t[i] = static_cast<T>(param.value);
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
