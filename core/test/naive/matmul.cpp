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
  NDArray a = F::from_list({1, 2, 3, 4, 5, 6}, {2, 3}, dtype::Int32());
  NDArray b = F::from_list({-1, 1, 2, 1, -1, 3}, {3, 2}, dtype::Int32());
  NDArray truth = F::from_list({0, 12, 0, 27}, {2, 2}, dtype::Int32());
  NDArray pred = F::empty({2, 2}, dtype::Int32());

  using Param = param::matmul;
  Param p;

  oprs.matmul(a, b, pred, p);

  print_data<int>(truth);
  print_data<int>(pred);

  assert_same_data<int>(pred, truth, 0.0001f);
}