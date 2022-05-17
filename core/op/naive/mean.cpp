#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_SPECIFIED_TYPE_INTERNAL(mean, nn_float64) {
  nn_size n = 1;
  nn_size src_idx[NN_MAX_NDIM], src_next[NN_MAX_NDIM],
      src_internal_idx[NN_MAX_NDIM], src_next_reverse[NN_MAX_NDIM];
  nn_size start_idx = 0;
  nn_size start_idx_reverse = 0;
  memset(src_idx, 0, sizeof src_idx);
  for (nn_size i = 0; i < linp.ndim; i++) {
    if (!param.dims[i]) {
      start_idx = i;
      break;
    }
  }
  for (nn_size i = 0; i < linp.ndim; i++) {
    if (param.dims[i]) {
      start_idx_reverse = i;
      break;
    }
  }
  for (nn_size i = 0; i < linp.ndim; i++) {
    if (!param.dims[i]) {
      src_next_reverse[i] = NN_MAX_NDIM;
      src_next[i] = 1;
      auto j = i + 1;
      while (j < linp.ndim && param.dims[j]) {
        src_next[i]++;
        j++;
      }
      n *= linp.shape[i];
    } else {
      src_next[i] = NN_MAX_NDIM;
      src_next_reverse[i] = 1;
      auto j = i + 1;
      while (j < linp.ndim && !param.dims[j]) {
        src_next_reverse[i]++;
        j++;
      }
    }
  }
  nn_size n_reverse = linp.total_elems() / n;

  auto increase_idx = [&]() {
    src_idx[start_idx]++;
    for (nn_size i = start_idx; i < linp.ndim; i += src_next[i]) {
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + src_next[i]]++;
      } else {
        break;
      }
    }
    return linp.indices_to_offset(src_idx);
  };

  auto increase_idx_internal = [&]() {
    src_internal_idx[start_idx_reverse]++;
    for (nn_size i = start_idx_reverse; i < linp.ndim;
         i += src_next_reverse[i]) {
      if (src_internal_idx[i] == linp.shape[i]) {
        src_internal_idx[i] = 0;
        src_internal_idx[i + src_next_reverse[i]]++;
      } else {
        break;
      }
    }
    return linp.indices_to_offset(src_internal_idx);
  };

  src_idx[start_idx] = -1;
  for (nn_size i = 0; i < n; i++) {
    auto base_offset = increase_idx();
    auto dst_offset = loup.indices_to_offset(src_idx);
    nn_float64 res = 0;
    memcpy(src_internal_idx, src_idx, sizeof(nn_size) * NN_MAX_NDIM);
    src_internal_idx[start_idx_reverse] = -1;
    for (nn_size j = 0; j < n_reverse; j++) {
      res += static_cast<nn_float64>(ptr_inp[increase_idx_internal()]);
    }
    ptr_oup[dst_offset] = res / n_reverse;
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
