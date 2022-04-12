#include <iostream>

#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(permute) {
  int duplicated = 0;
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    if (param.dims[i] >= inp.ndim) {
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "Invalid permute param.");
    }
    res[i] = inp[param.dims[i]];
    if (duplicated & (1 << param.dims[i])) {
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "Duplicated index in permute param.");
    }
    duplicated |= 1 << param.dims[i];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore