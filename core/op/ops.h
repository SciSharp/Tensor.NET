#pragma once

#include <string>

#include "core/base/include/ndArray.h"
#include "core/macro.h"
#include "core/op/param.h"

namespace nncore {
namespace opr {

using namespace param;

#define DEF_OP_METHOD_SINGLE_INPUT(_name)                    \
  virtual void _name(const NDArray& inp, const NDArray& oup, \
                     const param::_name& param) = 0;

#define DEF_OP_METHOD_DOUBLE_INPUT(_name)                                    \
  virtual void _name(const NDArray& a, const NDArray& b, const NDArray& oup, \
                     const param::_name& param) = 0;

#define NN_FOREACH_SINGLE_INPUT_OP(cb) cb(reshape) cb(transpose)

#define NN_FOREACH_DOUBLE_INPUT_OP(cb) cb(matmul) cb(dot)

#define NN_FOREACH_SINGLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(reshape, __VA_ARGS__) cb(transpose, __VA_ARGS__)

#define NN_FOREACH_DOUBLE_INPUT_OP_WITH_PARAM(cb, ...) \
  cb(matmul, __VA_ARGS__) cb(dot, __VA_ARGS__)

class OpBase {
 public:
  NN_FOREACH_SINGLE_INPUT_OP(DEF_OP_METHOD_SINGLE_INPUT)

  NN_FOREACH_DOUBLE_INPUT_OP(DEF_OP_METHOD_DOUBLE_INPUT)

  virtual ~OpBase() = default;
};

// #undef DEF_OP_METHOD_SINGLE_INPUT
// #undef DEF_OP_METHOD_DOUBLE_INPUT

}  // namespace opr

}  // namespace nncore
