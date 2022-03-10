#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {
template <typename T>
void OpNaiveImpl::reshape_internal(const NDArray& inp, const NDArray& oup,
                                   const Layout& linp, const Layout& loup,
                                   const param::reshape& param) {}

NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SINGLE_OUTPUT_OP_INTERNAL, OpNaiveImpl,
                            reshape)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
