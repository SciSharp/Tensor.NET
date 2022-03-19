#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(transpose) {
  if (param.dimA >= inp.ndim || param.dimB >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid permute param.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (int i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  res[inp.ndim - param.dimA - 1] = inp[inp.ndim - param.dimB - 1];
  res[inp.ndim - param.dimB - 1] = inp[inp.ndim - param.dimA - 1];
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore