#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(dot) {
  return Status(StatusCategory::NUMNET, StatusCode::NOT_IMPLEMENTED);
}

}  // namespace opr
}  // namespace nncore