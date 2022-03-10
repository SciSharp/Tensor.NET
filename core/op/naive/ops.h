#pragma once

#include "core/macro.h"
#include "core/op/common/ops.h"
#include "core/op/naive/macro.h"

namespace nncore {
namespace opr {
namespace naive {

class OpNaiveImpl final : public OpBase {
  NN_FOREACH_SINGLE_INPUT_OP(IMPL_OP_SINGLE_INPUT)

  NN_FOREACH_DOUBLE_INPUT_OP(IMPL_OP_DOUBLE_INPUT)
};

// NN_FOREACH_CTYPE_WITH_PARAM(EXPLICIT_DECLARE_TEMPLATE_CLASS, OpNaiveImpl)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
