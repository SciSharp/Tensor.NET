#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::max;

TEST(Naive, Max) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list(
      {38, 41, 15, 18, 29, 32, 14, 31, 13, 30, 45, 16, 34, 30, 16, 16, 36, 10,
       4,  46, 39, 27, 10, 11, 19, 31, 3,  10, 44, 27, 9,  7,  43, 1,  41, 14,
       31, 41, 12, 37, 4,  45, 9,  3,  2,  31, 28, 17, 38, 30, 20, 47, 42, 7,
       14, 40, 39, 33, 30, 41, 21, 37, 16, 4,  31, 26, 9,  7,  42, 40, 41, 42},
      {3, 4, 2, 3}, dtype::Int32());
  Tensor truth1 = F::from_list(
      {38, 46, 39, 30, 45, 32, 31, 41, 43, 37, 44, 45, 38, 40, 42, 47, 42, 42},
      {3, 1, 2, 3}, dtype::Int32());
  Param p1({false, true, false, false});

  Tensor pred1;
  ASSERT_TRUE(oprs->max(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);
}