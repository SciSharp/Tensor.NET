#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(matrix_inverse) {
  if (inp.ndim < 2) {
    return Status(
        StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
        "The tensor to calculate inverse must has at least two dims.");
  }
  if (inp.shape[inp.ndim - 1] != inp.shape[inp.ndim - 2]) {
    return Status(
        StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
        "The tensor to calculate inverse must has its last two dims square.");
  }
  if (!inp.is_contiguous()) {
    return Status(StatusCategory::NUMNET, StatusCode::FAIL,
                  "The tensor to calculate inverse must be contiguous.");
  }
  if (inp.dtype.enumv() != DTypeEnum::Float32 &&
      inp.dtype.enumv() != DTypeEnum::Float64) {
    return Status(StatusCategory::NUMNET, StatusCode::FAIL,
                  "The inverse op only support float and double type.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore