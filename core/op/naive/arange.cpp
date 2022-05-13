#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SELF_MODIFY_INTERNAL(arange) {
  nn_size n = layout.total_elems();
  auto value = param.start;
  for (nn_size i = 0; i < n; ++i) {
    t[i] = static_cast<T>(value);
    value += param.step;
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
