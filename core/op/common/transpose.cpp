#include "core/op/common/ops.h"

namespace nncore {
namespace opr {

IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(transpose) { return false; }

}  // namespace opr
}  // namespace nncore