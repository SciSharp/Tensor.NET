#include <iostream>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(permute) {
  size_t n = loup.total_elems();
  size_t src_idx[NN_MAX_NDIM], dst_idx[NN_MAX_NDIM];
  for (size_t i = 0; i < n; i++) {
    loup.offset_to_indices(i, dst_idx);
    for (int j = 0; j < loup.ndim; j++) {
      src_idx[j] = dst_idx[param.dims[j]];
    }
    size_t src_pos = linp.indices_to_offset(src_idx);
    ptr_oup[i] = ptr_inp[src_pos];
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
