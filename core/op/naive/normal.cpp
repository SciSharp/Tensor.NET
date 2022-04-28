#include <cmath>
#include <iostream>
#include <random>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SELF_MODIFY_INTERNAL(normal) {
  std::random_device rd;

  std::mt19937 e2(rd());

  std::normal_distribution<> dist(param.avg, param.std);
  nn_size idx_offset = layout.ndim - NN_MAX_NDIM;
  for (nn_size n = 0; n < (idx_offset == 0 ? layout[idx_offset] : 1); n++) {
    nn_size n_pos = n * layout.stride[idx_offset];
    for (nn_size c = 0; c < (idx_offset >= -1 ? layout[idx_offset + 1] : 1);
         c++) {
      nn_size nc_pos = c * layout.stride[idx_offset + 1] + n_pos;
      for (nn_size i = 0; i < (idx_offset >= -2 ? layout[idx_offset + 2] : 1);
           i++) {
        nn_size nch_pos = i * layout.stride[idx_offset + 2] + nc_pos;
        for (nn_size j = 0; j < layout[idx_offset + 3]; j++) {
          t[nch_pos + j * layout.stride[idx_offset + 3]] =
              static_cast<T>(dist(e2));
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
