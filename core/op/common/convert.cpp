#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

Status OpBase::deduce_layout_convert(Layout& inp, Layout& res,
                                     const param::convert& param) {
  if (param.target_type == DTypeEnum::Invalid) {
    return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                  "Invalid type convert param.");
  }
  res.dtype = DType::from_enum(param.target_type);
  res.ndim = inp.ndim;
  for (nn_size i = 0; i < inp.ndim; i++) {
    res[i] = inp[i];
  }
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore