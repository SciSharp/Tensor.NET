#pragma once

#include <string>

#include "core/base/include/status.h"
#include "core/base/include/tensor.h"
#include "core/op/common/param.h"

namespace nncore {
namespace opr {

using namespace param;

#define NN_FOREACH_SINGLE_INPUT_OP(cb) cb(reshape) cb(transpose) cb(permute)

#define NN_FOREACH_DOUBLE_INPUT_OP(cb) cb(matmul) cb(dot)

#define NN_FOREACH_SINGLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(reshape, __VA_ARGS__) cb(transpose, __VA_ARGS__)

#define NN_FOREACH_DOUBLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(matmul, __VA_ARGS__) cb(dot, __VA_ARGS__)

#define DEF_OP_SINGLE_INPUT(_name)                       \
 public:                                                 \
  virtual Status _name(const Tensor& inp, Tensor& oup,   \
                       const param::_name& param) = 0;   \
                                                         \
 protected:                                              \
  Status deduce_layout_##_name(Layout& inp, Layout& res, \
                               const param::_name param);

#define DEF_OP_DOUBLE_INPUT(_name)                                    \
 public:                                                              \
  virtual Status _name(const Tensor& a, const Tensor& b, Tensor& oup, \
                       const param::_name& param) = 0;                \
                                                                      \
 protected:                                                           \
  Status deduce_layout_##_name(Layout& a, Layout& b, Layout& res,     \
                               const param::_name param);

class OpBase {
  NN_FOREACH_SINGLE_INPUT_OP(DEF_OP_SINGLE_INPUT)

  NN_FOREACH_DOUBLE_INPUT_OP(DEF_OP_DOUBLE_INPUT)

  virtual ~OpBase() = default;
};

#undef DEF_OP_SINGLE_INPUT
#undef DEF_OP_DOUBLE_INPUT

#define IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(_name)                   \
  Status OpBase::deduce_layout_##_name(Layout& inp, Layout& res, \
                                       const param::_name param)
#define IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(_name)                            \
  Status OpBase::deduce_layout_##_name(Layout& a, Layout& b, Layout& res, \
                                       const param::_name param)

#define TYPE_SELECT_SINGLE_INPUT(_type, _name)                   \
  if (loup.dtype.is_ctype<_type>()) {                            \
    nn_return_status_if_error(_name##_internal<_type>(           \
        inp.ptr<_type>(), oup.ptr<_type>(), linp, loup, param)); \
    return Status::OK();                                         \
  }
#define TYPE_SELECT_DOUBLE_INPUT(_type, _name)                           \
  if (oup.layout.dtype.is_ctype<_type>()) {                              \
    nn_return_status_if_error(                                           \
        _name##_internal<_type>(a.ptr<_type>(), b.ptr<_type>(),          \
                                oup.ptr<_type>(), la, lb, loup, param)); \
    return Status::OK();                                                 \
  }

#define IMPL_OP_SINGLE_INPUT(_name)                                         \
 public:                                                                    \
  Status _name(const Tensor& inp, Tensor& oup, const param::_name& param) { \
    Layout linp(inp.layout);                                                \
    Layout loup;                                                            \
    nn_return_status_if_error(deduce_layout_##_name(linp, loup, param));    \
    loup.init_contiguous_stride();                                          \
    oup.relayout(loup);                                                     \
    NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT, _name)            \
    return Status::OK();                                                    \
  }                                                                         \
                                                                            \
 private:                                                                   \
  template <typename T>                                                     \
  Status _name##_internal(const T* inp, T* oup, const Layout& linp,         \
                          const Layout& loup, const param::_name& param);

#define IMPL_OP_DOUBLE_INPUT(_name)                                        \
 public:                                                                   \
  Status _name(const Tensor& a, const Tensor& b, Tensor& oup,              \
               const param::_name& param) {                                \
    Layout la(a.layout);                                                   \
    Layout lb(b.layout);                                                   \
    Layout loup;                                                           \
    nn_return_status_if_error(deduce_layout_##_name(la, lb, loup, param)); \
    loup.init_contiguous_stride();                                         \
    oup.relayout(loup);                                                    \
    NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_DOUBLE_INPUT, _name)           \
    return Status::OK();                                                   \
  }                                                                        \
                                                                           \
 private:                                                                  \
  template <typename T>                                                    \
  Status _name##_internal(const T* ptr_a, const T* ptr_b, T* ptr_oup,      \
                          const Layout& la, const Layout& lb,              \
                          const Layout& loup, const param::_name& param);

#define SPECIFY_SINGLE_OUTPUT_OP_INTERNAL(_type, _class_name, _op_name)     \
  template Status _class_name::_op_name##_internal<_type>(                  \
      const _type* inp, _type* oup, const Layout& linp, const Layout& loup, \
      const param::_op_name& param);

#define SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL(_type, _class_name, _op_name) \
  template Status _class_name::_op_name##_internal<_type>(              \
      const _type* ptr_a, const _type* ptr_b, _type* ptr_oup,           \
      const Layout& la, const Layout& lb, const Layout& loup,           \
      const param::_op_name& param);

// #undef DEF_OP_METHOD_SINGLE_INPUT
// #undef DEF_OP_METHOD_DOUBLE_INPUT

}  // namespace opr

}  // namespace nncore
