#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_SPECIFIED_TYPE_INTERNAL(argmxx, nn_int64) {
  nn_size n = loup.total_elems();
  nn_size src_idx[NN_MAX_NDIM], dst_idx[NN_MAX_NDIM];
  nn_size idx_offset[NN_MAX_NDIM];
  for (nn_size i = 0; i < linp.ndim; i++) {
    src_idx[i] = 0;
    idx_offset[i] = (i + 1) == param.axis ? 2 : 1;
  }
  src_idx[!param.axis] = -1;

  auto increase_idx = [&]() {
    src_idx[!param.axis]++;  // if axis = 0, increase src_idx[1]
    for (nn_size i = !param.axis; i < linp.ndim; i += idx_offset[i]) {
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + idx_offset[i]]++;
      }
    }
    return linp.indices_to_offset(src_idx);
  };

  for (nn_size i = 0; i < n; i++) {
    auto base_offset = increase_idx();
    T value = ptr_inp[base_offset];
    nn_int64 idx = 0;

    for (int i = 1; i < linp.shape[param.axis]; i++) {
      base_offset += linp.stride[param.axis];
      if (ptr_inp[base_offset] > value && param.is_max ||
          ptr_inp[base_offset] < value && !param.is_max) {
        value = ptr_inp[base_offset];
        idx = i;
      }
    }

    int temp = src_idx[param.axis];
    src_idx[param.axis] = 0;
    ptr_oup[loup.indices_to_offset(src_idx)] = idx;
    src_idx[param.axis] = temp;
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
