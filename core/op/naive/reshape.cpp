#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

using param = nncore::param::reshape;

template <typename T>
void OpNaiveImpl<T>::reshape(const NDArray& inp, const NDArray& oup,
                             const param& param) {}
}  // namespace naive
}  // namespace opr

}  // namespace nncore
