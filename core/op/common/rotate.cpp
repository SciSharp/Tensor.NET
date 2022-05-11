#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(rotate) {
  if (param.k <= 0 || param.k > 3) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The k must be between 1 and 3.");
  }
  if (param.dimA == param.dimB) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The dimA and dimB cannot be the same.");
  }
  if (inp.ndim <= 1) {
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "The tensor to be rotated must have at least two dims.");
  }
  if (param.dimA < 0 || param.dimB < 0 || param.dimA >= inp.ndim ||
      param.dimB >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The dimA or dimB exceeds the valid range.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  if (param.k != 2) {
    std::swap(res[param.dimA], res[param.dimB]);
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore