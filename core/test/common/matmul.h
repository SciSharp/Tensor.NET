#include <vector>

#include "core/op/param.h"

namespace nncore {
namespace test {
namespace matmul {
struct TestArgs {
  param::MatMul params;
  Shape a, b, oup;
  TestArgs(param::MatMul params, Shape a, Shape b, Shape oup)
      : params(params), a(a), b(b), oup(oup) {}
};

inline std::vector<TestArgs> get_args() {}

}  // namespace matmul

}  // namespace test

}  // namespace nncore
