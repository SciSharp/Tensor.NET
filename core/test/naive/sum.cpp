#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::sum;

TEST(Naive, Sum) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list(
      {19, 25, 12, 16, 22, 1,  37, 3,  22, 2,  27, 9,  21, 29, 13, 13, 10, 14,
       33, 44, 15, 6,  48, 25, 15, 37, 6,  30, 6,  14, 26, 0,  30, 4,  17, 7,
       26, 28, 28, 27, 42, 39, 38, 47, 25, 31, 34, 25, 11, 22, 26, 6,  47, 35,
       13, 46, 40, 21, 39, 18, 10, 10, 40, 40, 13, 31, 24, 11, 19, 31, 5,  16},
      {3, 4, 2, 3}, dtype::Int32());
  Tensor truth1 = F::from_list({273, 193, 306, 276, 272, 302}, {3, 1, 2, 1},
                               dtype::Int32());
  Param p1({false, true, false, true}, true);

  Tensor pred1;
  ASSERT_TRUE(oprs->sum(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list(
      {36, 33, 3,  25, 9,  19, 18, 45, 21, 41, 32, 19, 23, 10, 3,  17, 26, 33,
       37, 25, 44, 6,  7,  27, 46, 29, 42, 19, 42, 42, 0,  43, 4,  49, 27, 22,
       10, 2,  5,  21, 47, 39, 21, 32, 5,  1,  43, 40, 9,  30, 10, 29, 36, 46,
       47, 27, 36, 39, 26, 28, 5,  49, 21, 45, 45, 7,  44, 16, 44, 44, 1,  45},
      {3, 4, 2, 3}, dtype::Int32());
  Tensor truth2 = F::from_list({238, 267, 241, 283, 128, 280, 268, 214},
                               {1, 4, 2, 1}, dtype::Int32());
  Param p2({true, false, false, true}, true);

  Tensor pred2;
  ASSERT_TRUE(oprs->sum(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);

  // Group 3
  Tensor inp3 = F::from_list(
      {18, 4,  35, 22, 34, 18, 15, 5,  13, 49, 41, 3,  4,  35, 14, 22, 38, 29,
       42, 47, 30, 4,  9,  0,  35, 0,  43, 27, 44, 29, 2,  17, 16, 6,  36, 41,
       1,  22, 3,  30, 35, 13, 41, 45, 38, 44, 38, 21, 42, 41, 19, 24, 40, 39,
       37, 24, 28, 17, 22, 21, 15, 23, 35, 16, 31, 3,  29, 28, 0,  15, 17, 10},
      {3, 4, 2, 3}, dtype::Int32());
  Tensor truth3 = F::from_list({1734}, {1, 1, 1, 1}, dtype::Int32());
  Param p3({true, true, true, true}, true);

  Tensor pred3;
  ASSERT_TRUE(oprs->sum(inp3, pred3, p3).is_ok());
  assert_same_data<nn_int32>(pred3, truth3, 0.0001f);
}