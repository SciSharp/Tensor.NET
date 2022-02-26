#include "core/op/naive/ops.h"

#include "core/macro.h"
#include "core/op/naive/macro.h"

namespace nncore {
namespace opr {
namespace naive {

// #define DEF_MATCH_SINGLE_CONDITION(_type, _method) \
//   if (data_type.is_ctype<_type>()) {               \
//     _method<_type>(inp, oup, param);               \
//     return;                                        \
//   }

// #define DEF_MATCH_DOUBLE_CONDITION(_type, _method)                           \
//   nn_assert(a.layout.dtype.is_same_with(b.layout.dtype),                     \
//             "Different dtype of the inputs of MatMul, one is %s, the other " \
//             "is %s.\n",                                                      \
//             a.layout.dtype.name(), b.layout.dtype.name());                   \
//   nn_assert(a.layout.dtype.is_same_with(oup.layout.dtype),                   \
//             "Different dtype of input and output of MatMul, the "            \
//             "input is %s, "                                                  \
//             "the output is %s.\n",                                           \
//             a.layout.dtype.name(), oup.layout.dtype.name());                 \
//   if (data_type.is_ctype<_type>()) {                                         \
//     _method<_type>(a, b, oup, param);                                        \
//     return;                                                                  \
//   }

// #define DEF_NAIVE_SINGLE_INPUT_OP_IMPL(_name)                                 \
//   void OpNaiveImpl::_name(const NDArray &inp, const NDArray &oup,             \
//                           const param::_name &param) {                        \
//     DType data_type = inp.layout.dtype;                                       \
//     NN_FOREACH_CTYPE_WITH_PARAM(DEF_MATCH_SINGLE_CONDITION, _name##_internal) \
//     nn_throw("Invalid type: %s.", data_type.name());                          \
//   }

// #define DEF_NAIVE_DOUBLE_INPUT_OP_IMPL(_name)                                 \
//   void OpNaiveImpl::_name(const NDArray &a, const NDArray &b,                 \
//                           const NDArray &oup, const param::_name &param) {    \
//     DType data_type = a.layout.dtype;                                         \
//     NN_FOREACH_CTYPE_WITH_PARAM(DEF_MATCH_DOUBLE_CONDITION, _name##_internal) \
//     nn_throw("Invalid type: %s.", data_type.name());                          \
//   }

// NN_FOREACH_CTYPE_WITH_PARAM(EXPLICIT_DECLARE_TEMPLATE_CLASS, OpNaiveImpl)

// NN_FOREACH_SINGLE_INPUT_OP(DEF_NAIVE_SINGLE_INPUT_OP_IMPL)

// NN_FOREACH_DOUBLE_INPUT_OP(DEF_NAIVE_DOUBLE_INPUT_OP_IMPL)

// #undef DEF_NAIVE_SINGLE_INPUT_OP_IMPL
// #undef DEF_NAIVE_DOUBLE_INPUT_OP_IMPL

}  // namespace naive
}  // namespace opr

}  // namespace nncore
