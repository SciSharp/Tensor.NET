#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(flip) {
  nn_size n = loup.total_elems();
  nn_size src_idx[NN_MAX_NDIM];
  for (nn_size i = 0; i < n; i++) {
    loup.offset_to_indices(i, src_idx);
    ptr_oup[i] = -ptr_inp[linp.indices_to_offset(src_idx)];
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
