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
using Param = param::onehot;

TEST(Naive, Onehot) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 = F::from_list({2, 3, 4, 0}, {4}, dtype::Int32());
  Tensor truth1 =
      F::from_list({0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
                   {4, 5}, dtype::Int32());
  Param p1(4);

  Tensor pred1;
  ASSERT_TRUE(oprs->onehot(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);
}