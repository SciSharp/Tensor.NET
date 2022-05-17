#pragma once
#include <iostream>

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

  NN_FOREACH_SELF_MODIFY_OP(IMPL_OP_SELF_MODIFY)

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

 public:
  Status argmxx(const Tensor& inp, Tensor& oup, const param::argmxx& param) {
    Layout linp(inp.layout);
    if (oup.is_ptr_owner()) {
      Layout loup;
      nn_return_status_if_error(deduce_layout_argmxx(linp, loup, param));
      loup.init_contiguous_stride();
      oup.relayout(loup);
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT_SPECIFIED_TYPE,
                                  argmxx, loup, nn_int64)
    } else {
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT_SPECIFIED_TYPE,
                                  argmxx, oup.layout, nn_int64)
    }
    return Status::OK();
  }

  template <typename T>
  Status argmxx_internal(const T* inp, nn_int64* oup, const Layout& linp,
                         const Layout& loup, const param::argmxx& param);

 public:
  Status mean(const Tensor& inp, Tensor& oup, const param::mean& param) {
    Layout linp(inp.layout);
    if (oup.is_ptr_owner()) {
      Layout loup;
      nn_return_status_if_error(deduce_layout_mean(linp, loup, param));
      loup.init_contiguous_stride();
      oup.relayout(loup);
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT_SPECIFIED_TYPE, mean,
                                  loup, nn_float64)
    } else {
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT_SPECIFIED_TYPE, mean,
                                  oup.layout, nn_float64)
    }
    return Status::OK();
  }

  template <typename T>
  Status mean_internal(const T* inp, nn_float64* oup, const Layout& linp,
                       const Layout& loup, const param::mean& param);

 public:
  Status concat(const std::vector<const Tensor*>& inp, Tensor& oup,
                const param::concat& param) {
    if (oup.is_ptr_owner()) {
      Layout loup;
      nn_return_status_if_error(deduce_layout_concat(inp, loup, param));
      loup.init_contiguous_stride();
      oup.relayout(loup);
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_CONCAT, loup)
    } else {
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_CONCAT, oup.layout)
    }
    return Status::OK();
  }

  template <typename T>
  Status concat_internal(const std::vector<const Tensor*>& inp, T* oup,
                         const Layout& loup, const param::concat& param);
};

#define IMPL_NAIVE_SINGLE_INPUT_SPECIFIED_TYPE_INTERNAL(_name, _typeOup)    \
  NN_FOREACH_CTYPE_WITH_PARAM(                                              \
      SPECIFY_SINGLE_OUTPUT_SPECIFIED_TYPE_OP_INTERNAL, OpNaiveImpl, _name, \
      _typeOup)                                                             \
                                                                            \
  template <typename T>                                                     \
  Status OpNaiveImpl::_name##_internal(                                     \
      const T* ptr_inp, _typeOup* ptr_oup, const Layout& linp,              \
      const Layout& loup, [[maybe_unused]] const param::_name& param)

#define IMPL_NAIVE_SELF_MODIFY_INTERNAL(_name)                              \
  NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SELF_MODIFY_OP_INTERNAL, OpNaiveImpl, \
                              _name)                                        \
                                                                            \
  template <typename T>                                                     \
  Status OpNaiveImpl::_name##_internal(                                     \
      T* t, const Layout& layout, [[maybe_unused]] const param::_name& param)

#define IMPL_NAIVE_SINGLE_INPUT_INTERNAL(_name)                               \
  NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_SINGLE_OUTPUT_OP_INTERNAL, OpNaiveImpl, \
                              _name)                                          \
                                                                              \
  template <typename T>                                                       \
  Status OpNaiveImpl::_name##_internal(                                       \
      const T* ptr_inp, T* ptr_oup, const Layout& linp, const Layout& loup,   \
      [[maybe_unused]] const param::_name& param)

#define IMPL_NAIVE_DOUBLE_INPUT_INTERNAL(_name)                        \
  FOREACH_DOUBLE_INPUT_TYPE_PAIR(SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL,    \
                                 OpNaiveImpl, _name)                   \
                                                                       \
  template <typename TA, typename TB, typename TC>                     \
  Status OpNaiveImpl::_name##_internal(                                \
      const TA* ptr_a, const TB* ptr_b, TC* ptr_oup, const Layout& la, \
      const Layout& lb, const Layout& loup,                            \
      [[maybe_unused]] const param::_name& param)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
