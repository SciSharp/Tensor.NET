#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

FOREACH_DOUBLE_INPUT_TYPE_PAIR(SPECIFY_CONVERT_OP_INTERNAL, OpNaiveImpl)

template <typename TA, typename TB>
Status OpNaiveImpl::convert_internal(const TA* inp, TB* oup, const Layout& linp,
                                     const Layout& loup,
                                     const param::convert& param) {
  nn_size n = loup.total_elems();
  for (nn_size i = 0; i < n; i++) {
    oup[i] = static_cast<TB>(inp[i]);
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
