#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(negative) {
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore