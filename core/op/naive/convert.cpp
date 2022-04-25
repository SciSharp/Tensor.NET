#include <iostream>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

FOREACH_DOUBLE_INPUT_TYPE_PAIR(SPECIFY_CONVERT_OP_INTERNAL, OpNaiveImpl)

template <typename TA, typename TB>
Status OpNaiveImpl::convert_internal(const TA* inp, TB* oup, const Layout& linp,
                                     const Layout& loup,
                                     const param::convert& param) {
  nn_size idx_offset = loup.ndim - 4;
  for (nn_size n = 0; n < (idx_offset == 0 ? loup[idx_offset] : 1); n++) {
    nn_size n_offset_inp = n * linp.stride[idx_offset];
    nn_size n_offset_oup = n * loup.stride[idx_offset];
    for (nn_size c = 0; c < (idx_offset >= -1 ? loup[idx_offset + 1] : 1);
         c++) {
      nn_size nc_offset_inp = c * linp.stride[idx_offset + 1] + n_offset_inp;
      nn_size nc_offset_oup = c * loup.stride[idx_offset + 1] + n_offset_oup;
      for (nn_size i = 0; i < (idx_offset >= -2 ? loup[idx_offset + 2] : 1);
           i++) {
        nn_size nch_offset_inp =
            i * linp.stride[idx_offset + 2] + nc_offset_inp;
        nn_size nch_offset_oup =
            i * loup.stride[idx_offset + 2] + nc_offset_oup;
        for (nn_size j = 0; j < loup[idx_offset + 3]; j++) {
          nn_size inp_pos = nch_offset_inp + j * linp.stride[idx_offset + 3];
          nn_size oup_pos = nch_offset_oup + j * loup.stride[idx_offset + 3];
          oup[oup_pos] = static_cast<TB>(inp[inp_pos]);
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
