#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::flip;

TEST(Naive, Flip) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list({48, 43, 48, 0,  49, 4,  17, 40, 29, 16,
                              16, 49, 36, 14, 41, 31, 8,  26, 3,  45},
                             {4, 5}, dtype::Int32());
  Tensor truth1 = F::from_list({31, 8,  26, 3,  45, 16, 49, 36, 14, 41,
                                4,  17, 40, 29, 16, 48, 43, 48, 0,  49},
                               {4, 5}, dtype::Int32());
  Param p1({true, false, false, false});

  Tensor pred1;
  ASSERT_TRUE(oprs->flip(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list({25, 36, 9,  40, 42, 42, 12, 32, 8, 29, 37, 14,
                              13, 4,  44, 20, 19, 7,  12, 35, 5, 23, 25, 25},
                             {2, 3, 4}, dtype::Int32());
  Tensor truth2 = F::from_list({20, 44, 4,  13, 35, 12, 7,  19, 25, 25, 23, 5,
                                40, 9,  36, 25, 32, 12, 42, 42, 14, 37, 29, 8},
                               {2, 3, 4}, dtype::Int32());
  Param p2({true, false, true, false});

  Tensor pred2;
  ASSERT_TRUE(oprs->flip(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);

  // Group 2
  Tensor inp3 = F::from_list(
      {49, 8,  2,  7,  16, 40, 22, 5,  22, 4,  19, 20, 20, 44, 21, 47, 28, 21,
       18, 41, 38, 1,  34, 10, 9,  4,  7,  4,  11, 17, 45, 6,  0,  26, 22, 45,
       40, 18, 26, 39, 32, 27, 23, 31, 15, 37, 11, 26, 10, 27, 3,  37, 47, 41,
       23, 36, 45, 10, 31, 24, 33, 10, 8,  4,  34, 8,  32, 45, 25, 31, 28, 45,
       0,  39, 5,  38, 33, 19, 13, 3,  41, 3,  34, 7,  42, 28, 5,  7,  49, 39,
       13, 1,  32, 20, 28, 2,  39, 0,  9,  4,  17, 48, 7,  2,  21, 43, 41, 23,
       30, 20, 44, 13, 47, 49, 32, 30, 22, 40, 39, 2},
      {3, 4, 2, 5}, dtype::Int32());
  Tensor truth3 = F::from_list(
      {32, 49, 47, 13, 44, 2,  39, 40, 22, 30, 21, 2,  7,  48, 17, 20, 30, 23,
       41, 43, 28, 20, 32, 1,  13, 4,  9,  0,  39, 2,  42, 7,  34, 3,  41, 39,
       49, 7,  5,  28, 5,  39, 0,  45, 28, 3,  13, 19, 33, 38, 34, 4,  8,  10,
       33, 31, 25, 45, 32, 8,  23, 41, 47, 37, 3,  24, 31, 10, 45, 36, 15, 31,
       23, 27, 32, 27, 10, 26, 11, 37, 22, 26, 0,  6,  45, 39, 26, 18, 40, 45,
       9,  10, 34, 1,  38, 17, 11, 4,  7,  4,  21, 44, 20, 20, 19, 41, 18, 21,
       28, 47, 16, 7,  2,  8,  49, 4,  22, 5,  22, 40},
      {3, 4, 2, 5}, dtype::Int32());
  Param p3({true, true, false, true});

  Tensor pred3;
  ASSERT_TRUE(oprs->flip(inp3, pred3, p3).is_ok());
  assert_same_data<nn_int32>(pred3, truth3, 0.0001f);
}