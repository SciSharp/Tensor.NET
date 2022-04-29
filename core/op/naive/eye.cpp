#include <cmath>
#include <iostream>
#include <random>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SELF_MODIFY_INTERNAL(eye) {
  if (layout.ndim != 2) {
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "The eye could only be applied on  two-dimensional tensor.");
  }
  int m = layout.shape[0];
  int n = layout.shape[1];
  memset(t, 0, layout.total_elems() * layout.dtype.size());
  //  i + k >= 0     i >= -k i >= 0
  //  i + k < n      i < n-k i < m
  int k = param.k;
  int from = std::max(-k, 0);
  int to = std::min(n - k, m);
  for (int i = from; i < to; i++) {
    int j = i + k;
    t[i * n + j] = 1;
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
