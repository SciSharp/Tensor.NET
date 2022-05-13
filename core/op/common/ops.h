#pragma once

#include <string>
#include <vector>

#include "core/base/include/status.h"
#include "core/base/include/tensor.h"
#include "core/op/common/dtype_deduce.h"
#include "core/op/common/param.h"

namespace nncore {
namespace opr {

using namespace param;

#define NN_FOREACH_SELF_MODIFY_OP(cb) \
  cb(normal) cb(uniform) cb(eye) cb(fill) cb(linspace)

#define NN_FOREACH_SINGLE_INPUT_OP(cb)                                        \
  cb(transpose) cb(permute) cb(repeat) cb(flip) cb(matrix_inverse) cb(rotate) \
      cb(pad) cb(sort)

#define NN_FOREACH_DOUBLE_INPUT_OP(cb) cb(matmul) cb(dot) cb(boolindex)

#define NN_FOREACH_SINGLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(permute, __VA_ARGS__) cb(transpose, __VA_ARGS__)

#define NN_FOREACH_DOUBLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(matmul, __VA_ARGS__) cb(dot, __VA_ARGS__)

#define DEF_OP_SELF_MODIFY(_name) \
 public:                          \
  virtual Status _name(Tensor& t, const param::_name& param) = 0;

#define DEF_OP_SINGLE_INPUT(_name)                       \
 public:                                                 \
  virtual Status _name(const Tensor& inp, Tensor& oup,   \
                       const param::_name& param) = 0;   \
                                                         \
 protected:                                              \
  Status deduce_layout_##_name(Layout& inp, Layout& res, \
                               const param::_name& param);

#define DEF_OP_DOUBLE_INPUT(_name)                                    \
 public:                                                              \
  virtual Status _name(const Tensor& a, const Tensor& b, Tensor& oup, \
                       const param::_name& param) = 0;                \
                                                                      \
 protected:                                                           \
  Status deduce_layout_##_name(Layout& a, Layout& b, Layout& res,     \
                               const param::_name& param);

class OpBase {
  NN_FOREACH_SELF_MODIFY_OP(DEF_OP_SELF_MODIFY)

  NN_FOREACH_SINGLE_INPUT_OP(DEF_OP_SINGLE_INPUT)

  NN_FOREACH_DOUBLE_INPUT_OP(DEF_OP_DOUBLE_INPUT)

  // The convert op is a single input op but has type convert.
  // So we specially defined it manually here.
 public:
  virtual Status convert(const Tensor& inp, Tensor& oup,
                         const param::convert& param) = 0;

 protected:
  Status deduce_layout_convert(Layout& inp, Layout& res,
                               const param::convert& param);

 public:
  virtual Status argmxx(const Tensor& inp, Tensor& oup,
                        const param::argmxx& param) = 0;

 protected:
  Status deduce_layout_argmxx(Layout& inp, Layout& res,
                              const param::argmxx& param);

 public:
  virtual Status concat(const std::vector<const Tensor*>& inp, Tensor& oup,
                        const param::concat& param) = 0;

 protected:
  Status deduce_layout_concat(const std::vector<const Tensor*>& inp,
                              Layout& res, const param::concat& param);

  virtual ~OpBase() = default;
};

#undef DEF_OP_SINGLE_INPUT
#undef DEF_OP_DOUBLE_INPUT

#define IMPL_SINGLE_INPUT_LAYOUT_DEDUCE(_name)                   \
  Status OpBase::deduce_layout_##_name(Layout& inp, Layout& res, \
                                       const param::_name& param)
#define IMPL_DOUBLE_INPUT_LAYOUT_DEDUCE(_name)                            \
  Status OpBase::deduce_layout_##_name(Layout& a, Layout& b, Layout& res, \
                                       const param::_name& param)

#define IMPL_OP_SELF_MODIFY(_name)                                         \
 public:                                                                   \
  Status _name(Tensor& t, const param::_name& param) {                     \
    if (!t.is_mutable()) {                                                 \
      return Status(                                                       \
          StatusCategory::NUMNET, StatusCode::RUNTIME_EXCEPTION,           \
          "Self-modify op could only be used to create the Tensor, which " \
          "means the current tensor should be mutable.");                  \
    }                                                                      \
    NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SELF_MODIFY, _name)            \
    return Status::OK();                                                   \
  }                                                                        \
                                                                           \
  template <typename T>                                                    \
  Status _name##_internal(T* ptr, const Layout& layout,                    \
                          const param::_name& param);

// If the oup tensor is not the owner of the memory, we cannot deduce and
// relayout it. The responsibility of layout deduce belongs to the user. The
// main user is csharp api.
#define IMPL_OP_SINGLE_INPUT(_name)                                            \
 public:                                                                       \
  Status _name(const Tensor& inp, Tensor& oup, const param::_name& param) {    \
    Layout linp(inp.layout);                                                   \
    if (oup.is_ptr_owner()) {                                                  \
      Layout loup;                                                             \
      nn_return_status_if_error(deduce_layout_##_name(linp, loup, param));     \
      loup.init_contiguous_stride();                                           \
      oup.relayout(loup);                                                      \
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT, _name, loup)       \
    } else {                                                                   \
      NN_FOREACH_CTYPE_WITH_PARAM(TYPE_SELECT_SINGLE_INPUT, _name, oup.layout) \
    }                                                                          \
    return Status::OK();                                                       \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  Status _name##_internal(const T* inp, T* oup, const Layout& linp,            \
                          const Layout& loup, const param::_name& param);

#define IMPL_OP_DOUBLE_INPUT(_name)                                            \
 public:                                                                       \
  Status _name(const Tensor& a, const Tensor& b, Tensor& oup,                  \
               const param::_name& param) {                                    \
    Layout la(a.layout);                                                       \
    Layout lb(b.layout);                                                       \
    if (oup.is_ptr_owner()) {                                                  \
      Layout loup;                                                             \
      nn_return_status_if_error(deduce_layout_##_name(la, lb, loup, param));   \
      loup.init_contiguous_stride();                                           \
      oup.relayout(loup);                                                      \
      DOUBLE_INPUT_DTYPE_DEDUCE(la.dtype.enumv(), lb.dtype.enumv(), _name, la, \
                                lb, loup, param);                              \
    } else {                                                                   \
      DOUBLE_INPUT_DTYPE_DEDUCE(la.dtype.enumv(), lb.dtype.enumv(), _name, la, \
                                lb, oup.layout, param);                        \
    }                                                                          \
    return Status::OK();                                                       \
  }                                                                            \
                                                                               \
  template <typename TA, typename TB, typename TC>                             \
  Status _name##_internal(const TA* ptr_a, const TB* ptr_b, TC* ptr_oup,       \
                          const Layout& la, const Layout& lb,                  \
                          const Layout& loup, const param::_name& param);

}  // namespace opr

}  // namespace nncore
