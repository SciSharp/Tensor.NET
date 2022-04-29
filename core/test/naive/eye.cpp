#include "core/base/include/tensor.h"
#include "core/op/naive/ops.h"
#include "core/test/common/checker.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::eye;

TEST(Naive, Eye) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor t1 =
      F::from_list({116,  -50, 79,  -180, -28,  -96, -165, -27,  -2,  10,
                    -176, 113, -30, 122,  -116, -51, -48,  -142, 198, -108},
                   {4, 5}, dtype::Int32());
  Tensor truth1 =
      F::from_list({0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                   {4, 5}, dtype::Int32());
  Param p1(1);

  ASSERT_TRUE(oprs->eye(t1, p1).is_ok());
  assert_same_data<nn_int32>(t1, truth1, 0.0001f);

  // Group 2
  Tensor t2 =
      F::from_list({-149, 141, 6,    -5,   47,  -183, 95,  53,   -149, -129,
                    186,  99,  21,   84,   -14, 175,  -53, 62,   140,  59,
                    -14,  50,  -169, -189, 61,  -7,   129, -125, 120,  -109,
                    118,  162, -80,  58,   -91, 115,  127, 173,  -140, -95},
                   {8, 5}, dtype::Int32());
  Tensor truth2 =
      F::from_list({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                   {8, 5}, dtype::Int32());
  Param p2(-2);

  Tensor pred2;
  ASSERT_TRUE(oprs->eye(t2, p2).is_ok());
  assert_same_data<nn_int32>(t2, truth2, 0.0001f);
}