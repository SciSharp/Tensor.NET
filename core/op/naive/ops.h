#pragma once
#include "core/base/include/macro.h"
#include "core/op/common/ops.h"

namespace nncore {
namespace opr {
namespace naive {

class OpNaiveImpl final : public OpBase {
 private:
  OpNaiveImpl(){};
  ~OpNaiveImpl(){};
  OpNaiveImpl(const OpNaiveImpl&);
  OpNaiveImpl& operator=(const OpNaiveImpl&);

 public:
  static OpNaiveImpl* get_instance() {
    static OpNaiveImpl instance;
    return &instance;
  }

  NN_FOREACH_SINGLE_INPUT_OP(IMPL_OP_SINGLE_INPUT)

  NN_FOREACH_DOUBLE_INPUT_OP(IMPL_OP_DOUBLE_INPUT)

  // Define the implementation of convert op specially
 public:
  Status convert(const Tensor& inp, Tensor& oup, const param::convert& param) {
    Layout linp(inp.layout);
    if (oup.is_ptr_owner()) {
      Layout loup;
      nn_return_status_if_error(deduce_layout_convert(linp, loup, param));
      loup.init_contiguous_stride();
      oup.relayout(loup);
      TYPE_CONVERT_DEDUCE(linp.dtype.enumv(), loup.dtype.enumv(), linp, loup,
                          param);
    } else {
      TYPE_CONVERT_DEDUCE(linp.dtype.enumv(), oup.layout.dtype.enumv(), linp,
                          oup.layout, param);
    }
    return Status::OK();
  }

  template <typename TA, typename TB>
  Status convert_internal(const TA* inp, TB* oup, const Layout& linp,
                          const Layout& loup, const param::convert& param);
};

#define IMPL_NAIVE_SINGLE_INPUT_INTERNAL(_name)                                \
  NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SINGLE_OUTPUT_OP_INTERNAL, OpNaiveImpl,  \
                              _name)                                           \
                                                                               \
  template <typename T>                                                        \
  Status OpNaiveImpl::_name##_internal(const T* ptr_inp, T* ptr_oup,           \
                                       const Layout& linp, const Layout& loup, \
                                       const param::_name& param)

#define IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(_name)                        \
  FOREACH_DOUBLE_INPUT_TYPE_PAIR(SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL,    \
                                 OpNaiveImpl, _name)                   \
                                                                       \
  template <typename TA, typename TB, typename TC>                     \
  Status OpNaiveImpl::_name##_internal(                                \
      const TA* ptr_a, const TB* ptr_b, TC* ptr_oup, const Layout& la, \
      const Layout& lb, const Layout& loup, const param::_name& param)

// NN_FOREACH_CTYPE_WITH_PARAM(EXPLICIT_DECLARE_TEMPLATE_CLASS, OpNaiveImpl)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
