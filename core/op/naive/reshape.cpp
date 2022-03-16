#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {
IMPL_NAIVE_SINGLE_INPUT_INTERNAL(reshape) {
  return Status(StatusCategory::NUMNET, StatusCode::NOT_IMPLEMENTED);
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
