#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(repeat) {
  if (param.axis < 0 || param.axis >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid repeat axis param.");
  }
  if (param.repeats <= 0) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The repeat count must be positive number.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  res[param.axis] = inp[param.axis] * param.repeats;
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore