#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(onehot) {
  memset(ptr_oup, 0, sizeof(T) * loup.total_elems());
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
    src_idx[loup.ndim - 1] = ptr_inp[src_offset];
    if (src_idx[loup.ndim - 1] > param.max_val || src_idx[loup.ndim - 1] < 0) {
      return Status(StatusCategory::NUMNET, StatusCode::FAIL,
                    "Invalid data (" + std::to_string(src_idx[loup.ndim - 1]) +
                        ") in onehot.");
    }
    auto dst_offset = loup.indices_to_offset(src_idx);
    ptr_oup[dst_offset] = 1;
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
