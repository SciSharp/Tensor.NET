#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(dot) { return false; }

}  // namespace opr
}  // namespace nncore