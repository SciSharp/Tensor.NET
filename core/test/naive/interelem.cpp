#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::interelem;

TEST(Naive, Interelem) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor a1 = F::from_list({1, 2, 3, 4, 5, 6}, {2, 3}, dtype::Int32());
  Tensor b1 = F::from_list({-1, 1, 2, 1, -1, 3}, {2, 3}, dtype::Int32());
  Tensor truth1 = F::from_list({-1, 2, 6, 4, -5, 18}, {2, 3}, dtype::Int32());
  Param p1(Param::Operation::Mul);

  Tensor pred1;
  ASSERT_TRUE(oprs->interelem(a1, b1, pred1, p1).is_ok());
  assert_same_data<int>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor a2 = F::from_list({1, 3, 5, 7, 9, 0}, {2, 3}, dtype::Int32());
  Tensor b2 = F::from_list({1, 2, 3}, {3}, dtype::Int32());
  Tensor truth2 = F::from_list({1, 6, 15, 7, 18, 0}, {2, 3}, dtype::Int32());
  Param p2(Param::Operation::Mul);

  Tensor pred2;
  ASSERT_TRUE(oprs->interelem(a2, b2, pred2, p2).is_ok());
  assert_same_data<int>(pred2, truth2, 0.0001f);

  // Group 4
  Tensor a4 = F::from_list({1, 2, 5, -2, -4, 6, 1, 2, 5, -2, -4, 6,
                            1, 2, 5, -2, -4, 6, 1, 2, 5, -2, -4, 6},
                           {1, 4, 2, 3}, dtype::Int32());
  Tensor b4 = F::from_list({1, 1, 2, 3, -2, -4, 8, 15, -7, -1, 5, 0},
                           {1, 4, 1, 3}, dtype::Int32());
  Tensor truth4 =
      F::from_list({1, 2,  10,  -2,  -4,  12,  3,  -4, -20, -6, 8,   -24,
                    8, 30, -35, -16, -60, -42, -1, 10, 0,   2,  -20, 0},
                   {1, 4, 2, 3}, dtype::Int32());
  Param p4(Param::Operation::Mul);

  Tensor pred4;
  ASSERT_TRUE(oprs->interelem(a4, b4, pred4, p4).is_ok());
  assert_same_data<int>(pred4, truth4, 0.0001f);

  // Group 5
  Tensor a5 = F::from_list({1, 2, 35, 43, 5, 14}, {2, 3}, dtype::Int32());
  Tensor b5 = F::from_list({1, 2, 9, 5, 2, 4}, {2, 3}, dtype::Int32());
  Tensor truth5 = F::from_list({0, 0, 8, 3, 1, 2}, {2, 3}, dtype::Int32());
  Param p5(Param::Operation::Mod);

  Tensor pred5;
  ASSERT_TRUE(oprs->interelem(a5, b5, pred5, p5).is_ok());
  assert_same_data<int>(pred5, truth5, 0.0001f);

  // Group 5
  Tensor a6 = F::from_list({1, 2, 4}, {3}, dtype::Int32());
  Tensor b6 = F::from_list({1, 3, 1}, {3}, dtype::Int32());
  Tensor truth6 = F::from_list({0, 1, 5}, {3}, dtype::Int32());
  Param p6(Param::Operation::Xor);

  Tensor pred6;
  ASSERT_TRUE(oprs->interelem(a6, b6, pred6, p6).is_ok());
  assert_same_data<int>(pred6, truth6, 0.0001f);
}