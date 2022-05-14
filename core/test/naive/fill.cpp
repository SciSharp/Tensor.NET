#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::fill;

TEST(Naive, Fill) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor t1({2, 3}, dtype::Int32());
  Tensor truth1 =
      F::from_list({123, 123, 123, 123, 123, 123}, {2, 3}, dtype::Int32());
  Param p1(123);

  ASSERT_TRUE(oprs->fill(t1, p1).is_ok());
  assert_same_data<nn_int32>(t1, truth1, 0.0001f);

  // Group 2
  Tensor t2({3, 2, 1, 4}, dtype::Float64());
  Tensor truth2 = F::from_list(
      {1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
       1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52},
      {3, 2, 1, 4}, dtype::Float64());
  Param p2(1.52);

  Tensor pred2;
  ASSERT_TRUE(oprs->fill(t2, p2).is_ok());
  assert_same_data<nn_float64>(t2, truth2, 0.0001f);
}