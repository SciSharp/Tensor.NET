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

#define IMPL_NAIVE_SINGLE_INPUT_INTERNAL(_name)                               \
  NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SINGLE_OUTPUT_OP_INTERNAL, OpNaiveImpl, \
                              _name)                                          \
                                                                              \
  template <typename T>                                                       \
  void OpNaiveImpl::_name##_internal(const T* ptr_inp, T* ptr_oup,            \
                                     const Layout& linp, const Layout& loup,  \
                                     const param::_name& param)

#define IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(_name)                               \
  NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL, OpNaiveImpl, \
                              _name)                                          \
                                                                              \
  template <typename T>                                                       \
  void OpNaiveImpl::_name##_internal(                                         \
      const T* ptr_a, const T* ptr_b, T* ptr_oup, const Layout& la,           \
      const Layout& lb, const Layout& loup, const param::_name& param)

// NN_FOREACH_CTYPE_WITH_PARAM(EXPLICIT_DECLARE_TEMPLATE_CLASS, OpNaiveImpl)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
