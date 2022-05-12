#include "core/base/include/tensor.h"
#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::pad;

TEST(Naive, Pad) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list(
      {16, 30, 14, 28, 33, 22, 10, 11, 25, 34, 38, 48, 22, 20, 32, 34, 47, 16},
      {2, 3, 3}, dtype::Int32());
  Tensor truth1 = F::from_list(
      {5,  5,  3,  3, 3,  6,  5,  5,  1,  1, 1, 6,  5,  5,  1,  1, 1, 6,  5,
       5,  1,  1,  1, 6,  5,  5,  4,  4,  4, 6, 5,  5,  3,  3,  3, 6, 5,  5,
       16, 30, 14, 6, 5,  5,  28, 33, 22, 6, 5, 5,  10, 11, 25, 6, 5, 5,  4,
       4,  4,  6,  5, 5,  3,  3,  3,  6,  5, 5, 34, 38, 48, 6,  5, 5, 22, 20,
       32, 6,  5,  5, 34, 47, 16, 6,  5,  5, 4, 4,  4,  6,  5,  5, 3, 3,  3,
       6,  5,  5,  2, 2,  2,  6,  5,  5,  2, 2, 2,  6,  5,  5,  2, 2, 2,  6,
       5,  5,  4,  4, 4,  6,  5,  5,  3,  3, 3, 6,  5,  5,  2,  2, 2, 6,  5,
       5,  2,  2,  2, 6,  5,  5,  2,  2,  2, 6, 5,  5,  4,  4,  4, 6},
      {5, 5, 6}, dtype::Int32());
  Param p1(Param::Mode::Constant, 6, {1, 2, 1, 1, 2, 1}, {1, 2, 3, 4, 5, 6});

  Tensor pred1;
  ASSERT_TRUE(oprs->pad(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list({10, 17, 44, 19, 18, 29, 39, 13, 14, 13, 37, 35},
                             {2, 3, 2}, dtype::Int32());
  Tensor truth2 = F::from_list(
      {10, 10, 10, 17, 17, 17, 17, 10, 10, 10, 17, 17, 17, 17, 44, 44, 44, 19,
       19, 19, 19, 18, 18, 18, 29, 29, 29, 29, 18, 18, 18, 29, 29, 29, 29, 10,
       10, 10, 17, 17, 17, 17, 10, 10, 10, 17, 17, 17, 17, 44, 44, 44, 19, 19,
       19, 19, 18, 18, 18, 29, 29, 29, 29, 18, 18, 18, 29, 29, 29, 29, 39, 39,
       39, 13, 13, 13, 13, 39, 39, 39, 13, 13, 13, 13, 14, 14, 14, 13, 13, 13,
       13, 37, 37, 37, 35, 35, 35, 35, 37, 37, 37, 35, 35, 35, 35, 39, 39, 39,
       13, 13, 13, 13, 39, 39, 39, 13, 13, 13, 13, 14, 14, 14, 13, 13, 13, 13,
       37, 37, 37, 35, 35, 35, 35, 37, 37, 37, 35, 35, 35, 35, 39, 39, 39, 13,
       13, 13, 13, 39, 39, 39, 13, 13, 13, 13, 14, 14, 14, 13, 13, 13, 13, 37,
       37, 37, 35, 35, 35, 35, 37, 37, 37, 35, 35, 35, 35},
      {5, 5, 7}, dtype::Int32());
  Param p2(Param::Mode::Edge, 6, {1, 2, 1, 1, 2, 3}, {});

  Tensor pred2;
  ASSERT_TRUE(oprs->pad(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);

  // Group 3
  Tensor inp3 = F::from_list({27, 42, 27, 41, 26, 28}, {2, 3}, dtype::Int32());
  Tensor truth3 = F::from_list({42, 41, 42, 28, 42, 42, 27, 42, 27, 42,
                                41, 41, 26, 28, 41, 41, 41, 42, 28, 41},
                               {4, 5}, dtype::Int32());
  Param p3(Param::Mode::Maximum, 4, {1, 1, 1, 1}, {});

  Tensor pred3;
  ASSERT_TRUE(oprs->pad(inp3, pred3, p3).is_ok());
  assert_same_data<nn_int32>(pred3, truth3, 0.0001f);
}