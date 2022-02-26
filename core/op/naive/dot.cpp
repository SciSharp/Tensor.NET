#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

using param = nncore::param::dot;

template <typename T>
void OpNaiveImpl<T>::dot(const NDArray& a, const NDArray& b, const NDArray& oup,
                         const param& param) {}
}  // namespace naive
}  // namespace opr

}  // namespace nncore
