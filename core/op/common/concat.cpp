#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

Status OpBase::deduce_layout_concat(const std::vector<const Tensor*>& inp,
                                    Layout& res, const param::concat& param) {
  nn_size n = static_cast<nn_size>(inp.size());
  if (n < 2) {
    return Status(
        StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
        "At least two tensor should be inputted to execute the concat op.");
  }
  nn_size ndim = inp[0]->layout.ndim;
  nn_size size_on_axis = inp[0]->layout.shape[param.axis];
  for (nn_size i = 1; i < n; i++) {
    if (inp[i]->layout.ndim != ndim) {
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "The tensors to concat must have the same ndim.");
    }
    if (!inp[i]->layout.dtype.is_same_with(inp[0]->layout.dtype)) {
      return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                    "Tensors to concat must have the same dtype.");
    }
    for (nn_size j = 0; j < ndim; j++) {
      if (j != param.axis &&
          inp[i]->layout.shape[j] != inp[0]->layout.shape[j]) {
        return Status(StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
                      "Tensors to concat should have the same shape except the "
                      "specified axis.");
      }
    }
    size_on_axis += inp[i]->layout.shape[param.axis];
  }
  if (param.axis < 0 || param.axis >= ndim) {
    return Status(
        StatusCategory::NUMNET, StatusCode::INVALID_PARAM,
        "The specified axis to concat exceeds the max ndim of the tensors.");
  }
  res.dtype = inp[0]->layout.dtype;
  res.ndim = ndim;
  for (nn_size i = 0; i < ndim; i++) {
    res.shape[i] = inp[0]->layout.shape[i];
  }
  res.shape[param.axis] = size_on_axis;
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore