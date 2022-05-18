#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::min;

TEST(Naive, Min) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list(
      {26,  4,   43,  -6,  -10, 9,   6,   1,   -7,  29,  27,  -47, -6,  -6, -48,
       -39, -35, 41,  38,  -5,  43,  14,  40,  -49, 48,  -29, -7,  -27, 17, -41,
       42,  45,  45,  49,  -14, 20,  -38, -32, 2,   20,  -4,  37,  -3,  22, -10,
       -47, 2,   -7,  -42, -38, -36, -42, -16, 8,   33,  19,  -15, 4,   34, -41,
       -30, 29,  -44, 46,  45,  40,  3,   3,   -44, -25, -14, -15},
      {3, 4, 2, 3}, dtype::Int32());
  Tensor truth1 = F::from_list({-42, -42, -15, -47, -48, -39, -44, -49},
                               {1, 4, 2, 1}, dtype::Int32());
  Param p1({true, false, false, true});

  Tensor pred1;
  ASSERT_TRUE(oprs->min(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);
}