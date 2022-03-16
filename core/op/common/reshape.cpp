#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(reshape) {
  return Status(StatusCategory::NUMNET, StatusCode::NOT_IMPLEMENTED);
}

}  // namespace opr
}  // namespace nncore