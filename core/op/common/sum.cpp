#include <iostream>

#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(sum) {
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    if (param.dims[i]) {
      res.shape[i] = 1;
    } else {
      res.shape[i] = inp.shape[i];
    }
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore