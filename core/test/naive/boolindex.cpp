#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::boolindex;

TEST(Naive, BoolIndex) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor a = F::from_list(
      {39, 42, 16, 33, 44, 35, 16, 46, 0, 48, 3, 25, 12, 14, 25, 46, 42, 35},
      {2, 3, 3}, dtype::Int32());
  Tensor b = F::from_list({1, 0, 1, 0, 1, 0}, {2, 1, 3}, dtype::Bool());
  Tensor truth = F::from_list(
      {39, 0, 16, 33, 0, 35, 16, 0, 0, 0, 3, 0, 0, 14, 0, 0, 42, 0}, {2, 3, 3},
      dtype::Int32());
  Param p1;

  Tensor pred;
  ASSERT_TRUE(oprs->boolindex(a, b, pred, p1).is_ok());
  assert_same_data<int>(pred, truth, 0.0001f);
}