#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(matmul) {
  res.dtype = DType::from_enum(
      deduce_double_input_op(a.dtype.enumv(), b.dtype.enumv()));
  if (!a.ndim || !b.ndim)
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "Encountered empty tensor during deduce.");
  if (a.is_scalar() && b.is_scalar()) {
    res.shape[0] = 1;
    res.ndim = 1;
    return Status::OK();
  }
  if (a.ndim == 1 && b.ndim == 1) {
    res.shape[0] = a.shape[0];
    res.shape[1] = b.shape[0];
    res.ndim = 2;
    a.broadcast_inplace({a.shape[0], 1});
    b.broadcast_inplace({1, b.shape[0]});
    return Status::OK();
  }
  nn_size dim = a.ndim > b.ndim ? a.ndim : b.ndim;
  nn_size min_dim = a.ndim < b.ndim ? a.ndim : b.ndim;
  res.ndim = dim;
  Shape a_dst_shape(a);
  Shape b_dst_shape(b);

  for (nn_size i = 0; i < dim - 2; i++) {
    int a_idx = static_cast<int>(a.ndim - i - 3);
    int b_idx = static_cast<int>(b.ndim - i - 3);
    int target_idx = dim - i - 3;
    if (a_idx >= 0 &&
        (b_idx < 0 || a.shape[a_idx] == 1 || b.shape[b_idx] == 1 ||
         a.shape[a_idx] == b.shape[b_idx])) {
      if (b_idx < 0)
        res.shape[target_idx] = a.shape[a_idx];
      else
        res.shape[target_idx] =
            a.shape[a_idx] == 1 ? b.shape[b_idx] : a.shape[a_idx];
    } else if (a_idx < 0) {
      res.shape[target_idx] = b.shape[b_idx];
    } else {
      return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                    "Tensor shapes mismatched for matmul, a is " +
                        a.Shape::to_string() + ", b is " +
                        b.Shape::to_string());
    }
    a_dst_shape.shape[target_idx] = res.shape[target_idx];
    b_dst_shape.shape[target_idx] = res.shape[target_idx];
  }
  if (a.ndim == 1 && b.shape[b.ndim - 2] != a.shape[0] ||
      b.ndim == 1 && b.shape[0] != a.shape[a.ndim - 1] ||
      a.ndim != 1 && b.ndim != 1 &&
          a.shape[a.ndim - 1] != b.shape[b.ndim - 2]) {
    return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_SHAPE,
                  "Tensor shapes mismatched for matmul, a is " +
                      a.Shape::to_string() + ", b is " + b.Shape::to_string());
  }
  res.shape[dim - 2] = a_dst_shape[dim - 2] =
      a.ndim == 1 ? 1 : a.shape[a.ndim - 2];
  res.shape[dim - 1] = b_dst_shape[dim - 1] =
      b.ndim == 1 ? 1 : b.shape[b.ndim - 1];
  a_dst_shape[dim - 1] = b_dst_shape[dim - 2] = a.shape[a.ndim - 1];
  if (b.ndim == 1) {
    b.shape[1] = 1;
    b.ndim = 2;
    b.stride[0] = 1;
    b.stride[1] = 0;
  }
  a_dst_shape.ndim = b_dst_shape.ndim = res.ndim;
  a.broadcast_inplace(a_dst_shape);
  b.broadcast_inplace(b_dst_shape);
  return Status::OK();
}

}  // namespace opr
}  // namespace nncore