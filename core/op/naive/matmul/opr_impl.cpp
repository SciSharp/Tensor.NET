#include "core/op/naive/matmul/opr_impl.h"

namespace nncore {
namespace opr {
namespace naive {
template <typename T>
void MatMulImpl::exec_internal(const NDArray& a, const NDArray& b,
                               const NDArray& oup,
                               const MatMulBase::Param& param) {
  Shape shape_a(a.layout);
  nn_assert(shape_a.ndim > 0);
  if (shape_a.ndim == 1) {
    shape_a.shape[1] = shape_a.shape[0];
    shape_a.shape[0] = 1;
    shape_a.ndim++;
  }
  Shape shape_b(b.layout);
  nn_assert(shape_b.ndim > 0);
  if (shape_b.ndim == 1) {
    shape_b.shape[1] = shape_b.shape[0];
    shape_b.shape[0] = 1;
    shape_b.ndim++;
  }

  nn_assert(shape_a.shape[shape_a.ndim - 1] == shape_b.shape[shape_b.ndim - 2],
            "Invalid shape of the input of matmul, a is %s and b is %s.",
            shape_a.to_string().c_str(), shape_b.to_string().c_str());
  bool is_broadcast = false;
  bool bradcast_a = true;
  for (size_t i = shape_a.ndim - 3, j = shape_b.ndim - 3;; i--, j--) {
    if (i < 0 || j < 0) break;
    nn_assert(shape_a[i] == shape_b[j] || shape_a[i] == 1 || shape_b[j] == 1,
              "The input of matmul can not broadcast, a is %s and b is %s.",
              shape_a.to_string().c_str(), shape_b.to_string().c_str());
    if (shape_a[i] != shape_b[j] && (shape_a[i] == 1 || shape_b[j] == 1)) {
      is_broadcast = true;
      if (shape_a[i] > 1 && shape_b[j] == 1) bradcast_a = false;
    }
  }

  T* ptr_a = a.ptr<T>();
  T* ptr_b = b.ptr<T>();
  T* ptr_oup = oup.ptr<T>();

  if (!is_broadcast) {
    size_t nc = 0;
    size_t hw_a = shape_a[shape_a.ndim - 2] * shape_a[shape_a.ndim - 1];
    size_t hw_b = shape_b[shape_b.ndim - 2] * shape_b[shape_b.ndim - 1];
    size_t hw_oup =
        oup.layout[oup.layout.ndim - 2] * oup.layout[oup.layout.ndim - 1];
    for (size_t i = 0; i < shape_a.ndim - 2; i++) {
      nc += shape_a[i];
    }
    for (size_t p = 0; p < nc; p++) {
      for (size_t i = 0; i < shape_a[shape_a.ndim - 2]; i++) {
        for (size_t j = 0; j < shape_b[shape_b.ndim - 1]; j++) {
          T r = 0;
          for (size_t k = 0; k < shape_a[shape_a.ndim - 1]; k++) {
            size_t a_pos = p * hw_a + i * shape_a[shape_a.ndim - 1] + k;
            size_t b_pos = p * hw_b + k * shape_a[shape_a.ndim - 1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          size_t oup_pos = p * hw_oup + i * oup.layout[oup.layout.ndim - 1] + j;
          ptr_oup[oup_pos] = r;
        }
      }
    }
  } else if (bradcast_a) {
    size_t nc = 0;
    size_t hw_b = shape_b[shape_b.ndim - 2] * shape_b[shape_b.ndim - 1];
    size_t hw_oup =
        oup.layout[oup.layout.ndim - 2] * oup.layout[oup.layout.ndim - 1];
    for (size_t i = 0; i < shape_b.ndim - 2; i++) {
      nc += shape_b[i];
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
    size_t nc = 0;
    size_t hw_a = shape_a[shape_a.ndim - 2] * shape_a[shape_a.ndim - 1];
    size_t hw_oup =
        oup.layout[oup.layout.ndim - 2] * oup.layout[oup.layout.ndim - 1];
    for (size_t i = 0; i < shape_a.ndim - 2; i++) {
      nc += shape_a[i];
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
}
}  // namespace naive
}  // namespace opr

}  // namespace nncore
