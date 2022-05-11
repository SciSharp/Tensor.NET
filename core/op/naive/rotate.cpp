#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(rotate) {
  nn_size rotate_n = linp.shape[param.dimA] * linp.shape[param.dimB];
  nn_size other_n = linp.total_elems() / rotate_n;
  nn_size src_idx[NN_MAX_NDIM], src_next[NN_MAX_NDIM], dst_idx[NN_MAX_NDIM];
  memset(src_idx, 0, sizeof src_idx);
  nn_size start = NN_MAX_NDIM;
  for (nn_size i = 0; i < linp.ndim; i++) {
    if (i == param.dimA || i == param.dimB) {
      src_next[i] = NN_MAX_NDIM;
      continue;
    }
    if (start == NN_MAX_NDIM) start = i;
    if (i + 1 == param.dimA || i + 1 == param.dimB) {
      if (i + 2 == param.dimA || i + 2 == param.dimB) {
        src_next[i] = 3;
      } else {
        src_next[i] = 2;
      }
    } else {
      src_next[i] = 1;
    }
  }

  auto increase_idx = [&]() {
    src_idx[start]++;
    for (nn_size i = start; i < linp.ndim; i += src_next[i]) {
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + src_next[i]]++;
      }
    }
  };

  auto src_to_dst = [&]() {
    if (param.k == 1) {
      dst_idx[param.dimA] = src_idx[param.dimB];
      dst_idx[param.dimB] = linp.shape[param.dimA] - src_idx[param.dimA] - 1;
    } else if (param.k == 2) {
      dst_idx[param.dimA] = linp.shape[param.dimA] - src_idx[param.dimA] - 1;
      dst_idx[param.dimB] = linp.shape[param.dimB] - src_idx[param.dimB] - 1;
    } else {
      dst_idx[param.dimB] = src_idx[param.dimA];
      dst_idx[param.dimA] = linp.shape[param.dimB] - src_idx[param.dimB] - 1;
    }
  };

  if (start != NN_MAX_NDIM) {
    src_idx[start] = -1;
  }
  for (nn_size i = 0; i < other_n; i++) {
    increase_idx();
    for (nn_size j = 0; j < linp.shape[param.dimA]; j++) {
      src_idx[param.dimA] = j;
      for (nn_size k = 0; k < linp.shape[param.dimB]; k++) {
        src_idx[param.dimB] = k;
        auto src_offset = linp.indices_to_offset(src_idx);
        memcpy(dst_idx, src_idx, sizeof(nn_size) * NN_MAX_NDIM);
        src_to_dst();
        auto dst_offset = loup.indices_to_offset(dst_idx);
        ptr_oup[dst_offset] = ptr_inp[src_offset];
      }
    }
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
