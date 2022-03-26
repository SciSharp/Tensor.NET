#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(matmul) {
  res.dtype = a.dtype;
  if (!a.ndim || !b.ndim)
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "Encountered empty tensor during deduce.");
  if (a.is_scalar() && b.is_scalar()) {
    res.shape[0] = 1;
    res.ndim = 1;
    return Status::OK();
  }
  if (a.ndim == 1 && b.ndim == 1) {
    res.shape[0] = b.shape[0];
    res.shape[1] = a.shape[0];
    res.ndim = 2;
    a.broadcast_inplace({a.shape[0], 1});
    b.broadcast_inplace({1, b.shape[0]});
    return Status::OK();
  }
  int dim = a.ndim > b.ndim ? a.ndim : b.ndim;
  res.ndim = dim;
  std::vector<size_t> a_dst_shape;
  std::vector<size_t> b_dst_shape;
  for (int i = dim - 1; i >= 2; i--) {
    if (i >= a.ndim || i < b.ndim && a.shape[i] == 1) {
      res.shape[i] = b.shape[i];
    } else if (i >= b.ndim || i < a.ndim && b.shape[i] == 1) {
      res.shape[i] = a.shape[i];
    } else if (a.shape[i] == b.shape[i]) {
      res.shape[i] = a.shape[i];
    } else {
      return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                    "Tensor shapes mismatched for matmul, a is " +
                        a.Shape::to_string() + ", b is " +
                        b.Shape::to_string());
    }
    a_dst_shape.push_back(res.shape[i]);
    b_dst_shape.push_back(res.shape[i]);
  }
  if (a.ndim == 1 && b.shape[1] != a.shape[0] ||
      b.ndim == 1 && b.shape[0] != a.shape[0] ||
      a.ndim != 1 && b.ndim != 1 && a.shape[0] != b.shape[1]) {
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "Tensor shapes mismatched for matmul, a is " +
                      a.Shape::to_string() + ", b is " + b.Shape::to_string());
  }
  res.shape[1] = a.ndim == 1 ? 1 : a.shape[1];
  res.shape[0] = b.ndim == 1 ? 1 : b.shape[0];
  a_dst_shape.push_back(a.ndim > 1 ? a.shape[1] : 1);
  b_dst_shape.push_back(b.ndim > 1 ? b.shape[1] : b.shape[0]);
  a_dst_shape.push_back(a.shape[0]);
  b_dst_shape.push_back(b.ndim > 1 ? b.shape[0] : 1);
  if (b.ndim == 1) {
    b[1] = b[0];
    b[0] = 1;
    b.ndim = 2;
    b.stride[1] = 1;
    b.stride[0] = 0;
  }
  a.broadcast_inplace(a_dst_shape);
  b.broadcast_inplace(b_dst_shape);
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore