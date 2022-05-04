#include <cmath>
#include <iostream>
#include <random>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SELF_MODIFY_INTERNAL(linspace) {
  nn_size n = layout.total_elems();
  auto step = (param.stop - param.start) /
              std::max(static_cast<double>(param.is_endpoint ? n - 1 : n), 1.0);
  for (nn_size i = 0; i < n; ++i) {
    t[i] = static_cast<T>(param.start + i * step);
  }
  return Status::OK();
}

} // namespace naive
} // namespace opr

} // namespace nncore
