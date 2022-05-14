#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::repeat;

TEST(Naive, Repeat) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list({8, 35, 28, 42, 8, 26, 43, 43, 0, 14, 9, 32},
                             {3, 4}, dtype::Int32());
  Tensor truth1 = F::from_list({8,  8,  35, 35, 28, 28, 42, 42, 8, 8, 26, 26,
                                43, 43, 43, 43, 0,  0,  14, 14, 9, 9, 32, 32},
                               {3, 8}, dtype::Int32());
  Param p1(2, 1);

  Tensor pred1;
  ASSERT_TRUE(oprs->repeat(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list(
      {31, 4, 2,  29, 20, 14, 45, 33, 16, 48, 49, 18, 33, 4,  4,  7,
       1,  8, 23, 41, 49, 16, 22, 1,  0,  11, 23, 20, 36, 35, 21, 14,
       42, 1, 1,  25, 27, 17, 8,  45, 25, 23, 11, 27, 4,  17, 29, 14},
      {2, 3, 4, 2}, dtype::Int32());
  Tensor truth2 = F::from_list(
      {31, 4,  2,  29, 20, 14, 45, 33, 31, 4,  2,  29, 20, 14, 45, 33, 31, 4,
       2,  29, 20, 14, 45, 33, 16, 48, 49, 18, 33, 4,  4,  7,  16, 48, 49, 18,
       33, 4,  4,  7,  16, 48, 49, 18, 33, 4,  4,  7,  1,  8,  23, 41, 49, 16,
       22, 1,  1,  8,  23, 41, 49, 16, 22, 1,  1,  8,  23, 41, 49, 16, 22, 1,
       0,  11, 23, 20, 36, 35, 21, 14, 0,  11, 23, 20, 36, 35, 21, 14, 0,  11,
       23, 20, 36, 35, 21, 14, 42, 1,  1,  25, 27, 17, 8,  45, 42, 1,  1,  25,
       27, 17, 8,  45, 42, 1,  1,  25, 27, 17, 8,  45, 25, 23, 11, 27, 4,  17,
       29, 14, 25, 23, 11, 27, 4,  17, 29, 14, 25, 23, 11, 27, 4,  17, 29, 14},
      {2, 9, 4, 2}, dtype::Int32());
  Param p2(3, 1);

  Tensor pred2;
  ASSERT_TRUE(oprs->repeat(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);
}