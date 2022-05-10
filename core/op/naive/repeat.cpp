#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(repeat) {
  nn_size n = linp.total_elems();
  nn_size src_idx[NN_MAX_NDIM];
  memset(src_idx, 0, sizeof src_idx);
  src_idx[0] = -1;

  auto increase_idx = [&]() {
    src_idx[0]++;
    for (nn_size i = 0; i < linp.ndim; i++) {
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + 1]++;
      }
    }
    return linp.indices_to_offset(src_idx);
  };

  for (nn_size i = 0; i < n; i++) {
    auto src_offset = increase_idx();
    T value = ptr_inp[src_offset];

    auto temp = src_idx[param.axis];
    src_idx[param.axis] *= param.repeats;

    auto dst_offset = loup.indices_to_offset(src_idx);

    for (nn_size j = 0; j < param.repeats; j++) {
      ptr_oup[dst_offset] = value;
      dst_offset += loup.stride[param.axis];
    }

    src_idx[param.axis] = temp;
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
