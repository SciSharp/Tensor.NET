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
using Param = param::convert;

TEST(Naive, Convert) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 =
      F::from_list({116,  0,   79,  0,   -28,  -96, -165, -27,  -2,  10,
                    -176, 113, -30, 122, -116, -51, -48,  -142, 198, 0},
                   {4, 5}, dtype::Int32());
  Tensor truth1 = F::from_list(
      {true, false, true, false, true, true, true, true, true, true,
       true, true,  true, true,  true, true, true, true, true, false},
      {4, 5}, dtype::Bool());
  Param p1(DTypeEnum::Bool);

  Tensor pred1;
  ASSERT_TRUE(oprs->convert(inp1, pred1, p1).is_ok());
  assert_same_data<bool>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list(
      {-149, 141, 6,    -5,   47,   -183, 95,  53,   -149, -129, 186,  99,
       21,   84,  -14,  175,  -53,  62,   140, 59,   -14,  50,   -169, -189,
       61,   -7,  129,  -125, 120,  -109, 118, 162,  -80,  58,   -91,  115,
       127,  173, -140, -95,  0,    192,  66,  -83,  -188, -29,  -92,  -145,
       -6,   6,   -149, 159,  155,  93,   -74, -147, -37,  182,  123,  -49,
       112,  -23, -157, -73,  -182, -120, -45, 31,   -119, -2,   18,   -199},
      {3, 4, 6}, dtype::Int32());
  Tensor truth2 = F::from_list(
      {-149, 141, 6,    -5,   47,   -183, 95,  53,   -149, -129, 186,  99,
       21,   84,  -14,  175,  -53,  62,   140, 59,   -14,  50,   -169, -189,
       61,   -7,  129,  -125, 120,  -109, 118, 162,  -80,  58,   -91,  115,
       127,  173, -140, -95,  0,    192,  66,  -83,  -188, -29,  -92,  -145,
       -6,   6,   -149, 159,  155,  93,   -74, -147, -37,  182,  123,  -49,
       112,  -23, -157, -73,  -182, -120, -45, 31,   -119, -2,   18,   -199},
      {3, 4, 6}, dtype::Float64());
  Param p2(DTypeEnum::Float64);

  Tensor pred2;
  ASSERT_TRUE(oprs->convert(inp2, pred2, p2).is_ok());
  assert_same_data<nn_float64>(pred2, truth2, 0.0001f);
}