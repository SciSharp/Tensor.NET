#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::mean;

TEST(Naive, Mean) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list(
      {31, 5,  34, 48, 1,  41, 29, 44, 21, 39, 5,  40, 29, 24, 24, 6,
       13, 43, 10, 12, 33, 19, 46, 22, 24, 45, 15, 12, 7,  3,  2,  12,
       14, 19, 23, 48, 0,  10, 42, 15, 10, 8,  11, 18, 23, 11, 37, 9},
      {2, 4, 3, 2}, dtype::Int32());
  Tensor truth1 =
      F::from_list({23.0, 28.583333333333332, 17.333333333333332, 17.5},
                   {2, 1, 1, 2}, dtype::Float64());
  Param p1({false, true, true, false}, true);

  Tensor pred1;
  ASSERT_TRUE(oprs->mean(inp1, pred1, p1).is_ok());
  assert_same_data<nn_float64>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list(
      {25, 37, 43, 28, 10, 27, 4,  25, 26, 14, 36, 18, 41, 47, 22, 27,
       16, 38, 28, 29, 43, 4,  30, 36, 40, 6,  35, 1,  35, 1,  47, 48,
       7,  2,  35, 28, 26, 37, 49, 46, 7,  37, 30, 38, 12, 32, 9,  21},
      {2, 4, 3, 2}, dtype::Int32());
  Tensor truth2 = F::from_list({24.0, 24.166666666666668, 32.75, 26.0},
                               {1, 4, 1, 1}, dtype::Float64());
  Param p2({true, false, true, true}, true);

  Tensor pred2;
  ASSERT_TRUE(oprs->mean(inp2, pred2, p2).is_ok());
  assert_same_data<nn_float64>(pred2, truth2, 0.0001f);
}