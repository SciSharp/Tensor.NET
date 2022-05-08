#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {
IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(boolindex) {
  nn_size n = loup.total_elems();
  nn_size target_shape[4];
  for (nn_size i = 0; i < n; i++) {
    loup.offset_to_indices(i, target_shape);
    auto bool_offset = lb.indices_to_offset(target_shape);
    auto src_offset = la.indices_to_offset(target_shape);
    ptr_oup[i] = ptr_b[bool_offset] ? ptr_a[src_offset] : static_cast<TA>(0);
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
