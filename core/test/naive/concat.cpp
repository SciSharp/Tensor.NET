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
using Param = param::concat;

TEST(Naive, Concat) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor a = F::from_list(
      {39, 42, 16, 33, 44, 35, 16, 46, 0, 48, 3, 25, 12, 14, 25, 46, 42, 35},
      {2, 3, 3}, dtype::Int32());
  Tensor b = F::from_list({6, 26, 0, 12, 7, 40, 41, 1, 19, 25, 10, 32},
                          {2, 2, 3}, dtype::Int32());
  Tensor c = F::from_list({37, 0,  0,  41, 9,  28, 47, 16, 4, 47, 15, 39,
                           29, 12, 20, 12, 29, 0,  49, 40, 9, 48, 42, 6},
                          {2, 4, 3}, dtype::Int32());
  Tensor truth = F::from_list(
      {39, 42, 16, 33, 44, 35, 16, 46, 0,  6,  26, 0,  12, 7,  40, 37, 0,  0,
       41, 9,  28, 47, 16, 4,  47, 15, 39, 48, 3,  25, 12, 14, 25, 46, 42, 35,
       41, 1,  19, 25, 10, 32, 29, 12, 20, 12, 29, 0,  49, 40, 9,  48, 42, 6},
      {2, 9, 3}, dtype::Int32());
  Param p1(1);

  Tensor pred;
  ASSERT_TRUE(oprs->concat({&a, &b, &c}, pred, p1).is_ok());
  assert_same_data<int>(pred, truth, 0.0001f);
}