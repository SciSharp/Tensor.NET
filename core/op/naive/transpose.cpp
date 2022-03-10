#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

template <typename T>
void OpNaiveImpl::transpose_internal(const NDArray& inp, const NDArray& oup,
                                     const Layout& linp, const Layout& loup,
                                     const param::transpose& param) {}

NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SINGLE_OUTPUT_OP_INTERNAL, OpNaiveImpl,
                            transpose)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
