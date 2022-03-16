#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {
IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(dot) {
  return Status(StatusCategory::NUMNET, StatusCode::NOT_IMPLEMENTED);
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
