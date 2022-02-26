#pragma once

#include "core/macro.h"
#include "core/op/naive/macro.h"
#include "core/op/ops.h"

namespace nncore {
namespace opr {
namespace naive {

// #define DEF_NAIVE_SINGLE_INPUT_OP_DECLARE(_name)                \
//  protected:                                                     \
//   void _name##_internal(const NDArray &inp, const NDArray &oup, \
//                         const param::##_name &param);           \
//                                                                 \
//  public:                                                        \
//   void _name(const NDArray &inp, const NDArray &oup,            \
//              const param::##_name &param);

// #define DEF_NAIVE_DOUBLE_INPUT_OP_DECLARE(_name)                          \
//  protected:                                                               \
//   void _name##_internal(const NDArray &a, const NDArray &b,               \
//                         const NDArray &oup, const param::##_name &param); \
//                                                                           \
//  public:                                                                  \
//   void _name(const NDArray &a, const NDArray &b, const NDArray &oup,      \
//              const param::##_name &param);

#define DEF_NAIVE_SINGLE_INPUT_OP_DECLARE(_name) \
  void _name(const NDArray &inp, const NDArray &oup, const param::_name &param);

#define DEF_NAIVE_DOUBLE_INPUT_OP_DECLARE(_name)                     \
  void _name(const NDArray &a, const NDArray &b, const NDArray &oup, \
             const param::_name &param);

template <typename T>
class OpNaiveImpl final : public OpBase {
 public:
  NN_FOREACH_SINGLE_INPUT_OP(DEF_NAIVE_SINGLE_INPUT_OP_DECLARE)

  NN_FOREACH_DOUBLE_INPUT_OP(DEF_NAIVE_DOUBLE_INPUT_OP_DECLARE)
};

#undef DEF_NAIVE_SINGLE_INPUT_OP_DECLARE
#undef DEF_NAIVE_DOUBLE_INPUT_OP_DECLARE

NN_FOREACH_CTYPE_WITH_PARAM(EXPLICIT_DECLARE_TEMPLATE_CLASS, OpNaiveImpl)

}  // namespace naive
}  // namespace opr

}  // namespace nncore
