#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(matmul) {
  res.dtype = a.dtype;
  res.format = a.format;
  if (!a.ndim || !b.ndim) return false;
  if (a.is_scalar() && b.is_scalar()) {
    res.shape[0] = 1;
    res.ndim = 1;
    return true;
  }
  if (a.ndim == 1 && b.ndim == 1) {
    res.shape[0] = b.shape[0];
    res.shape[1] = a.shape[0];
    res.ndim = 2;
    a.self_broadcast({a.shape[0], 1});
    b.self_broadcast({1, b.shape[0]});
    return true;
  }
  size_t dim = a.ndim > b.ndim ? a.ndim : b.ndim;
  res.ndim = dim;
  std::vector<size_t> a_dst_shape;
  std::vector<size_t> b_dst_shape;
  for (size_t i = dim - 1; i >= 2; i--) {
    if (i >= a.ndim || i < b.ndim && a.shape[i] == 1) {
      res.shape[i] = b.shape[i];
    } else if (i >= b.ndim || i < a.ndim && b.shape[i] == 1) {
      res.shape[i] = a.shape[i];
    } else if (a.shape[i] == b.shape[i]) {
      res.shape[i] = a.shape[i];
    } else {
      return false;
    }
    a_dst_shape.push_back(res.shape[i]);
    b_dst_shape.push_back(res.shape[i]);
  }
  if (a.ndim == 1 && b.shape[1] != a.shape[0] ||
      b.ndim == 1 && b.shape[0] != a.shape[0] ||
      a.ndim != 1 && b.ndim != 1 && a.shape[0] != b.shape[1]) {
    return false;
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
  a.self_broadcast(a_dst_shape);
  b.self_broadcast(b_dst_shape);
  return true;
}

}  // namespace opr
}  // namespace nncore