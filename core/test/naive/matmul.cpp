#include "core/test/common/matmul.h"

#include "core/base/include/ndArray.h"
#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;

TEST(Naive, Matmul) {
  OpNaiveImpl<int> oprs;

  // Group 1
  NDArray a1 = F::from_list({1, 2, 3, 4, 5, 6}, {2, 3}, dtype::Int32());
  NDArray b1 = F::from_list({-1, 1, 2, 1, -1, 3}, {3, 2}, dtype::Int32());
  NDArray truth1 = F::from_list({0, 12, 0, 27}, {2, 2}, dtype::Int32());
  NDArray pred1 = F::empty({2, 2}, dtype::Int32());

  using Param = param::matmul;
  Param p1;

  oprs.matmul(a1, b1, pred1, p1);
  assert_same_data<int>(pred1, truth1, 0.0001f);

  // Group 2
  NDArray a2 = F::from_list({1, 3, 5, 7, 9}, {5, 1}, dtype::Int32());
  NDArray b2 = F::from_list({2, 4, 6, 8, 10}, {1, 5}, dtype::Int32());
  NDArray truth2 =
      F::from_list({2,  4,  6,  8,  10, 6,  12, 18, 24, 30, 10, 20, 30,
                    40, 50, 14, 28, 42, 56, 70, 18, 36, 54, 72, 90},
                   {5, 5}, dtype::Int32());
  NDArray pred2 = F::empty({5, 5}, dtype::Int32());

  using Param = param::matmul;
  Param p2;

  oprs.matmul(a2, b2, pred2, p2);

  assert_same_data<int>(pred2, truth2, 0.0001f);

  // Group 3
  NDArray a3 = F::from_list({1, 3, 5, 7, 9}, {5}, dtype::Int32());
  NDArray b3 = F::from_list({2, 4, 6, 8, 10}, {5}, dtype::Int32());
  NDArray truth3 = F::from_list({190}, {1}, dtype::Int32());
  NDArray pred3 = F::empty({1}, dtype::Int32());

  using Param = param::matmul;
  Param p3;

  oprs.matmul(a3, b3, pred3, p3);

  assert_same_data<int>(pred3, truth3, 0.0001f);

  // Group 4
  NDArray a4 = F::from_list({1, 2, 5, -2, -4, 6}, {1, 1, 2, 3}, dtype::Int32());
  NDArray b4 = F::from_list({1, 1, 2, 3, -2, -4, 8, 15, -7, -1, 5, 0},
                            {4, 3, 1}, dtype::Int32());
  NDArray truth4 = F::from_list({13, 6, -21, -22, 3, -118, 9, -18},
                                {1, 4, 2, 1}, dtype::Int32());
  NDArray pred4 = F::empty({1, 4, 2, 1}, dtype::Int32());

  using Param = param::matmul;
  Param p4;

  oprs.matmul(a4, b4, pred4, p4);

  assert_same_data<int>(pred4, truth4, 0.0001f);
}