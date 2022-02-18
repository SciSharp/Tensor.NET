#include "core/test/common/matmul.h"

#include "core/base/include/ndArray.h"
#include "core/op/naive/matmul/opr_impl.h"
#include "core/test/common/factory.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;

using F = NDArrayFactory;

TEST(Naive, Matmul) {
  // Group 1
  NDArray a = F::from_list({1, 2, 3, 4, 5, 6}, {2, 3}, dtype::Int32());
  NDArray b = F::from_list({-1, 1, 2, 1, -1, 3}, {3, 2}, dtype::Int32());
  NDArray truth = F::from_list({0, 12, 0, 27}, {2, 2}, dtype::Int32());
  NDArray pred = F::empty({2, 2}, dtype::Int32());

  using Param = param::MatMul;
  Param p;

  doMatMul(a, b, pred, p);

  ASSERT_TRUE(pred.equal<int>(truth));
}