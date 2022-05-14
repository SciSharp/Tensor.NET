#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::rotate;

TEST(Naive, Rotate) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list({36, 47, 38, 32, 0,  21, 37, 0, 6,  37,
                              5,  38, 33, 0,  15, 20, 12, 6, 10, 40},
                             {4, 5}, dtype::Int32());
  Tensor truth1 = F::from_list({20, 5,  21, 36, 12, 38, 37, 47, 6,  33,
                                0,  38, 10, 0,  6,  32, 40, 15, 37, 0},
                               {5, 4}, dtype::Int32());
  Param p1(1, 0, 1);

  Tensor pred1;
  ASSERT_TRUE(oprs->rotate(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list({39, 13, 44, 29, 1, 12, 5, 34, 33, 12, 22, 42,
                              34, 3,  10, 17, 0, 41, 4, 8,  48, 36, 21, 47},
                             {3, 2, 4}, dtype::Int32());
  Tensor truth2 = F::from_list({8,  4,  41, 0,  47, 21, 36, 48, 42, 22, 12, 33,
                                17, 10, 3,  34, 29, 44, 13, 39, 34, 5,  12, 1},
                               {3, 2, 4}, dtype::Int32());
  Param p2(2, 0, 2);

  Tensor pred2;
  ASSERT_TRUE(oprs->rotate(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);

  // Group 3
  Tensor inp3 = F::from_list(
      {35, 22, 31, 18, 20, 19, 11, 28, 10, 35, 6,  14, 1,  15, 48, 38, 41, 38,
       17, 29, 31, 35, 49, 47, 49, 4,  17, 41, 18, 48, 3,  27, 16, 35, 33, 28,
       30, 1,  9,  43, 16, 34, 18, 23, 35, 49, 40, 18, 19, 4,  23, 8,  44, 19,
       18, 25, 14, 21, 37, 42, 28, 16, 29, 44, 16, 3,  37, 14, 49, 46, 32, 33,
       13, 12, 28, 21, 22, 11, 7,  42, 33, 34, 10, 41, 0,  12, 43, 5,  18, 35,
       47, 0,  1,  30, 45, 4,  49, 12, 9,  2,  13, 14, 42, 20, 8,  28, 46, 31,
       10, 34, 19, 34, 42, 39, 10, 22, 36, 42, 20, 49},
      {3, 2, 4, 5}, dtype::Int32());
  Tensor truth3 = F::from_list(
      {38, 41, 38, 17, 29, 28, 30, 1,  9,  43, 6,  14, 1,  15, 48, 3,  27, 16,
       35, 33, 19, 11, 28, 10, 35, 4,  17, 41, 18, 48, 35, 22, 31, 18, 20, 31,
       35, 49, 47, 49, 25, 14, 21, 37, 42, 21, 22, 11, 7,  42, 23, 8,  44, 19,
       18, 32, 33, 13, 12, 28, 49, 40, 18, 19, 4,  3,  37, 14, 49, 46, 16, 34,
       18, 23, 35, 28, 16, 29, 44, 16, 4,  49, 12, 9,  2,  22, 36, 42, 20, 49,
       47, 0,  1,  30, 45, 19, 34, 42, 39, 10, 12, 43, 5,  18, 35, 28, 46, 31,
       10, 34, 33, 34, 10, 41, 0,  13, 14, 42, 20, 8},
      {3, 4, 2, 5}, dtype::Int32());
  Param p3(3, 1, 2);

  Tensor pred3;
  ASSERT_TRUE(oprs->rotate(inp3, pred3, p3).is_ok());
  assert_same_data<nn_int32>(pred3, truth3, 0.0001f);
}