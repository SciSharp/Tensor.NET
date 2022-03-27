#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(matmul) {
  nn_size n_total = 1;
  nn_size c_total = 1;
  if (loup.ndim >= 4) n_total = loup[3];
  if (loup.ndim >= 3) c_total = loup[2];
  for (nn_size n = 0; n < n_total; n++) {
    for (nn_size c = 0; c < c_total; c++) {
      for (nn_size i = 0; i < la[1]; i++) {
        for (nn_size j = 0; j < lb[0]; j++) {
          TC r = TC(0);
          for (nn_size k = 0; k < la[0]; k++) {
            nn_size a_pos =
                n * la.stride[3] + c * la.stride[2] + i * la.stride[1] + k;
            nn_size b_pos =
                n * lb.stride[3] + c * lb.stride[2] + k * lb.stride[1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          nn_size oup_pos =
              n * loup.stride[3] + c * loup.stride[2] + i * loup.stride[1] + j;
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
