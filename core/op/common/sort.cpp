#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(sort) {
  if (param.axis < 0 || param.axis >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid axis param for sort op.");
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