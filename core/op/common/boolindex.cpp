#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(boolindex) {
  if (a.ndim < b.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "Cannot broadcast bool index to the shape of target tensor.");
  }
  for (nn_size i = 0, j = 0; i < a.ndim && j < b.ndim; i++, j++) {
    nn_size a_idx = a.ndim - i - 1;
    nn_size b_idx = b.ndim - j - 1;
    if (a.shape[a_idx] != b.shape[b_idx] && a.shape[a_idx] != 1 &&
        b.shape[b_idx] != 1) {
      return Status(
          StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
          "Cannot broadcast bool index to the shape of target tensor.");
    }
  }
  b.broadcast_inplace(a);

  res = Layout(a);
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore