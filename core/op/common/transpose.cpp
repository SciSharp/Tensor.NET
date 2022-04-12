#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(transpose) {
  if (param.dimA >= inp.ndim || param.dimB >= inp.ndim) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid transpose param.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  res[param.dimA] = inp[param.dimB];
  res[param.dimB] = inp[param.dimA];
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore