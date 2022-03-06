#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

using param = nncore::param::matmul;

template <typename T>
void OpNaiveImpl<T>::matmul(const NDArray& a, const NDArray& b,
                            const NDArray& oup, const param& param) {
  Shape shape_a(a.layout);
  nn_assert(shape_a.ndim > 0);
  if (shape_a.ndim == 1) {
    shape_a.shape[1] = 1;
    shape_a.ndim++;
  }
  Shape shape_b(b.layout);
  nn_assert(shape_b.ndim > 0);
  if (shape_b.ndim == 1) {
    shape_b.shape[1] = shape_b.shape[0];
    shape_b.shape[0] = 1;
    shape_b.ndim++;
  }

  size_t min_ndim = shape_a.ndim > shape_b.ndim ? shape_b.ndim : shape_a.ndim;
  size_t max_ndim = shape_a.ndim > shape_b.ndim ? shape_a.ndim : shape_b.ndim;
  for (int i = max_ndim - 1; i >= shape_a.ndim; i--) {
    shape_a.shape[i] = 1;
    shape_a.ndim++;
  }
  for (int i = max_ndim - 1; i >= shape_b.ndim; i--) {
    shape_b.shape[i] = 1;
    shape_b.ndim++;
  }

  nn_assert(shape_a.shape[0] == shape_b.shape[1],
            "Invalid shape of the input of matmul, "
            "a is %s and b is %s.",
            shape_a.to_string().c_str(), shape_b.to_string().c_str());
  bool is_broadcast = false;
  bool broadcast_a = true;
  for (size_t i = min_ndim - 1; i >= 2; i--) {
    nn_assert(shape_a[i] == shape_b[i] || shape_a[i] == 1 || shape_b[i] == 1,
              "The input of matmul can not broadcast, a is %s and b is %s.",
              shape_a.to_string().c_str(), shape_b.to_string().c_str());
    if (shape_a[i] != shape_b[i] && (shape_a[i] == 1 || shape_b[i] == 1)) {
      is_broadcast = true;
      if (shape_a[i] > 1 && shape_b[i] == 1) broadcast_a = false;
    }
    // is_broadcast = true;
    // if (shape_a[i] > 1 && shape_b[i] == 1) broadcast_a = false;
  }

  T* ptr_a = a.ptr<T>();
  T* ptr_b = b.ptr<T>();
  T* ptr_oup = oup.ptr<T>();

  if (!is_broadcast) {
    size_t hw_a = shape_a[0] * shape_a[1];
    size_t hw_b = shape_b[0] * shape_b[1];
    size_t hw_oup = oup.layout[0] * oup.layout[1];
    size_t nc = 1;
    for (size_t i = 2; i < shape_a.ndim; i++) {
      nc *= shape_a[i];
    }
    for (size_t p = 0; p < nc; p++) {
      for (size_t i = 0; i < shape_a[1]; i++) {
        for (size_t j = 0; j < shape_b[0]; j++) {
          T r = T(0);
          for (size_t k = 0; k < shape_a[0]; k++) {
            size_t a_pos = p * hw_a + i * a.layout.stride[1] + k;
            size_t b_pos = p * hw_b + k * b.layout.stride[1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          size_t oup_pos = p * hw_oup + i * oup.layout.stride[1] + j;
          ptr_oup[oup_pos] = r;
        }
      }
    }
  } else if (broadcast_a) {
    size_t nc = 1;
    size_t hw_b = shape_b[shape_b.ndim - 2] * shape_b[shape_b.ndim - 1];
    size_t hw_oup =
        oup.layout[oup.layout.ndim - 2] * oup.layout[oup.layout.ndim - 1];
    for (size_t i = 2; i < shape_b.ndim - 2; i++) {
      nc *= shape_b[i];
    }
    for (size_t p = 0; p < nc; p++) {
      for (size_t i = 0; i < shape_a[shape_a.ndim - 2]; i++) {
        for (size_t j = 0; j < shape_b[shape_b.ndim - 1]; j++) {
          T r = 0;
          for (size_t k = 0; k < shape_a[shape_a.ndim - 1]; k++) {
            size_t a_pos = i * shape_a[shape_a.ndim - 1] + k;
            size_t b_pos = p * hw_b + k * shape_a[shape_a.ndim - 1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          size_t oup_pos = p * hw_oup + i * oup.layout[oup.layout.ndim - 1] + j;
          ptr_oup[oup_pos] = r;
        }
      }
    }
  } else {
    size_t nc = 1;
    size_t hw_a = shape_a[shape_a.ndim - 2] * shape_a[shape_a.ndim - 1];
    size_t hw_oup =
        oup.layout[oup.layout.ndim - 2] * oup.layout[oup.layout.ndim - 1];
    for (size_t i = 2; i < shape_a.ndim - 2; i++) {
      nc *= shape_a[i];
    }
    for (size_t p = 0; p < nc; p++) {
      for (size_t i = 0; i < shape_a[shape_a.ndim - 2]; i++) {
        for (size_t j = 0; j < shape_b[shape_b.ndim - 1]; j++) {
          T r = 0;
          for (size_t k = 0; k < shape_a[shape_a.ndim - 1]; k++) {
            size_t a_pos = p * hw_a + i * shape_a[shape_a.ndim - 1] + k;
            size_t b_pos = k * shape_a[shape_a.ndim - 1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          size_t oup_pos = p * hw_oup + i * oup.layout[oup.layout.ndim - 1] + j;
          ptr_oup[oup_pos] = r;
        }
      }
    }
  }
  int nnn = 0;
}
}  // namespace naive
}  // namespace opr

}  // namespace nncore
