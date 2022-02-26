#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

using param = nncore::param::transpose;

template <typename T>
void OpNaiveImpl<T>::transpose(const NDArray& inp, const NDArray& oup,
                               const param& param) {}
}  // namespace naive
}  // namespace opr

}  // namespace nncore
