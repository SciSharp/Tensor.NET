#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(matmul) {
  nn_size n_total = 1;
  nn_size c_total = 1;
  nn_size n_idx = 0;
  nn_size c_idx = loup.ndim == 4 ? 1 : 0;
  nn_size pre_idx = la.ndim - 2;
  nn_size a0 = la.ndim >= 2 ? la.shape[la.ndim - 2] : 1,
          a1 = la.shape[la.ndim - 1];
  nn_size b1 = lb.shape[lb.ndim - 1];
  if (loup.ndim == 4) {
    n_total = loup[n_idx];
    c_total = loup[c_idx];
  } else if (loup.ndim == 3)
    c_total = loup[c_idx];
  for (nn_size n = 0; n < n_total; n++) {
    nn_size n_offset_a = n * la.stride[n_idx];
    nn_size n_offset_b = n * lb.stride[n_idx];
    nn_size n_offset_oup = n * loup.stride[n_idx];
    for (nn_size c = 0; c < c_total; c++) {
      nn_size nc_offset_a = c * la.stride[c_idx] + n_offset_a;
      nn_size nc_offset_b = c * lb.stride[c_idx] + n_offset_b;
      nn_size nc_offset_oup = c * loup.stride[c_idx] + n_offset_oup;
      for (nn_size i = 0; i < a0; i++) {
        for (nn_size j = 0; j < b1; j++) {
          TC r = TC(0);
          for (nn_size k = 0; k < a1; k++) {
            nn_size a_pos = nc_offset_a + i * la.stride[pre_idx] + k;
            nn_size b_pos = nc_offset_b + k * lb.stride[pre_idx] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          nn_size oup_pos = nc_offset_oup + i * loup.stride[pre_idx] + j;
          ptr_oup[oup_pos] = r;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
