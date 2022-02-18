#pragma once

#include "core/macro.h"
#include "core/op/naive/macro.h"
#include "core/op/ops.h"

namespace nncore {
namespace opr {
namespace naive {

class MatMulImpl final : public MatMulBase {
  DEF_OP_IMPL_CTOR(MatMulImpl)

 protected:
  /*!
   * We assume the shape is valid here
   */
  template <typename T>
  void exec_internal(const NDArray &a, const NDArray &b, const NDArray &oup,
                     const MatMulBase::Param &param);

 public:
  void exec(const NDArray &a, const NDArray &b, const NDArray &oup,
            const MatMulBase::Param &param) override {
    DType data_type = a.layout.dtype;
    nn_assert(a.layout.dtype.is_same_with(b.layout.dtype),
              "Different dtype of the inputs of MatMul, one is %s, the other "
              "is %s.\n",
              a.layout.dtype.name(), b.layout.dtype.name());
    nn_assert(a.layout.dtype.is_same_with(oup.layout.dtype),
              "Different dtype of input and output of MatMul, the input is %s, "
              "the output is %s.\n",
              a.layout.dtype.name(), oup.layout.dtype.name());

#define DEF_CONDITION(_type)                \
  if (data_type.is_ctype<_type>()) {        \
    exec_internal<_type>(a, b, oup, param); \
    return;                                 \
  }

    FOREACH_OPR_TYPE_CHECK(DEF_CONDITION)
#undef DEF_CONDITION

    nn_throw("Invalid type: %s.", data_type.name());
  }
};
}  // namespace naive
}  // namespace opr

}  // namespace nncore
