#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {
template <typename T>
void OpNaiveImpl::dot_internal(const NDArray& a, const NDArray& b,
                               const NDArray& oup, const Layout& la,
                               const Layout& lb, const Layout& loup,
                               const param::dot& param) {}

NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL, OpNaiveImpl, dot)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
