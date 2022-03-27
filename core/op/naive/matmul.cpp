#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(matmul) {
  size_t n_total = 1;
  size_t c_total = 1;
  if (loup.ndim >= 4) n_total = loup[3];
  if (loup.ndim >= 3) c_total = loup[2];
  for (size_t n = 0; n < n_total; n++) {
    for (size_t c = 0; c < c_total; c++) {
      for (size_t i = 0; i < la[1]; i++) {
        for (size_t j = 0; j < lb[0]; j++) {
          TC r = TC(0);
          for (size_t k = 0; k < la[0]; k++) {
            size_t a_pos =
                n * la.stride[3] + c * la.stride[2] + i * la.stride[1] + k;
            size_t b_pos =
                n * lb.stride[3] + c * lb.stride[2] + k * lb.stride[1] + j;
            r += ptr_a[a_pos] * ptr_b[b_pos];
          }
          size_t oup_pos =
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
