#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(flip) {
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

  auto increase_without_axis = [&](nn_size axis) {
    src_idx[!axis]++;  // if axis = 0, increase src_idx[1]
    for (nn_size i = !axis; i < linp.ndim; i += (i + 1) == axis ? 2 : 1) {
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + ((i + 1) == axis ? 2 : 1)]++;
      }
    }
    return loup.indices_to_offset(src_idx);
  };

  for (nn_size i = 0; i < n; i++) {
    auto src_offset = increase_idx();
    auto dst_offset = loup.indices_to_offset(src_idx);
    ptr_oup[dst_offset] = ptr_inp[src_offset];
  }

  for (nn_size i = 0; i < linp.ndim; i++) {
    if (!param.dims[i]) continue;
    auto cnt = n / loup.shape[i];
    memset(src_idx, 0, sizeof src_idx);
    src_idx[!i] = -1;
    for (nn_size j = 0; j < cnt; j++) {
      auto dst_offset = increase_without_axis(i);
      for (nn_size k = 0; k < loup.shape[i] / 2; k++) {
        std::swap(
            ptr_oup[dst_offset + k * loup.stride[i]],
            ptr_oup[dst_offset + (loup.shape[i] - k - 1) * loup.stride[i]]);
      }
    }
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
