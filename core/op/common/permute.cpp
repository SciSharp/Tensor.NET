#include <iostream>

#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(permute) {
  int duplicated = 0;
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (int i = 0; i < inp.ndim; i++) {
    if (params.dims[i] >= inp.ndim) {
      std::cout << "dim" << std::endl;
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "Invalid permute param.");
    }
    res[params.dims[i]] = inp[i];
    if (duplicated & (1 << params.dims[i])) {
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "Duplicated index in permute param.");
    }
    duplicated += 1 << params.dims[i];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore