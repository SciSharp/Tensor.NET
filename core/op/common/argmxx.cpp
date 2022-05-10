#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(argmxx) {
  if (param.axis < 0 || param.axis >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid argmxx axis param.");
  }
  res.dtype = dtype::Int64();
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  res[param.axis] = 1;
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore