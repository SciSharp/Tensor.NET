#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(pad) {
  if (!param.size) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The size of width must be larger than 0.");
  }
  if (param.size != inp.ndim * 2) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "The dims to pad must have length 2 * ndim.");
  }
  res.dtype = inp.dtype;
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i] + param.width[i * 2] + param.width[i * 2 + 1];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore