#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(onehot) {
  res.dtype = inp.dtype;
  res.ndim = inp.ndim + 1;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  res[res.ndim - 1] = param.max_val + 1;
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore