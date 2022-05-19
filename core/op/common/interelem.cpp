#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(interelem) {
  auto dim = std::max(a.ndim, b.ndim);
  for (nn_size i = 0, j = 0, k = 0; i < a.ndim && j < b.ndim; i++, j++, k++) {
    nn_size a_idx = a.ndim - i - 1;
    nn_size b_idx = b.ndim - j - 1;
    if (a.shape[a_idx] != b.shape[b_idx] && a.shape[a_idx] != 1 &&
        b.shape[b_idx] != 1) {
      return Status(
          StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
          "Cannot broadcast bool index to the shape of target tensor.");
    } else if (a.shape[a_idx] == b.shape[b_idx]) {
      res.shape[dim - k - 1] = a.shape[a_idx];
    } else if (a.shape[a_idx] == 1) {
      res.shape[dim - k - 1] = b.shape[b_idx];
    } else if (b.shape[a_idx] == 1) {
      res.shape[dim - k - 1] = a.shape[a_idx];
    } else {
      return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                    "Unknown error when deducing the layout.");
    }
  }
  for (int i = std::min(a.ndim, b.ndim); i < a.ndim; i++) {
    res.shape[dim - i - 1] = a.shape[a.ndim - i - 1];
  }
  for (int i = std::min(a.ndim, b.ndim); i < b.ndim; i++) {
    res.shape[dim - i - 1] = b.shape[b.ndim - i - 1];
  }
  res.dtype = DType::from_enum(
      deduce_double_input_op(a.dtype.enumv(), b.dtype.enumv()));
  res.ndim = dim;
  a.broadcast_inplace(res);
  b.broadcast_inplace(res);
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore