#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(matmul){
  size_t hw_a = la[0] * la[1];
  size_t hw_b = lb[0] * lb[1];
  size_t hw_oup = loup[0] * loup[1];
  size_t nc = 1;
  for (size_t i = 2; i < la.ndim; i++) {
    nc *= la[i];
  }
  for (size_t p = 0; p < nc; p++) {
    for (size_t i = 0; i < la[1]; i++) {
      for (size_t j = 0; j < lb[0]; j++) {
        T r = T(0);
        for (size_t k = 0; k < la[0]; k++) {
          size_t a_pos = p * hw_a + i * la.stride[1] + k;
          size_t b_pos = p * hw_b + k * lb.stride[1] + j;
          r += ptr_a[a_pos] * ptr_b[b_pos];
        }
        size_t oup_pos = p * hw_oup + i * loup.stride[1] + j;
        ptr_oup[oup_pos] = r;
      }
    }
  }
}


}  // namespace naive
}  // namespace opr

}  // namespace nncore
