#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::argmxx;

TEST(Naive, Argmxx) {
  OpBase* oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor inp1 =
      F::from_list({33, 11, 13, 48, 14, 49, 14, 34, 43, 19, 13, 12, 0,  15, 29,
                    1,  28, 6,  2,  20, 31, 39, 47, 34, 40, 12, 7,  7,  24, 23,
                    20, 43, 30, 35, 45, 1,  15, 44, 30, 9,  25, 42, 21, 12, 41,
                    41, 8,  45, 5,  43, 30, 42, 2,  13, 18, 46, 7,  34, 2,  20},
                   {3, 4, 5}, dtype::Int32());
  Tensor truth1 = F::from_list({1, 3, 1, 0, 2, 0, 2, 0, 2, 2, 3, 0, 1, 2, 1},
                               {3, 1, 5}, dtype::Int64());
  Param p1(1, true);

  Tensor pred1;
  ASSERT_TRUE(oprs->argmxx(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int64>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list({10, 9,  14, 14, 41, 36, 39, 17, 39, 23,
                              48, 35, 44, 23, 45, 49, 42, 27, 23, 4},
                             {4, 5}, dtype::Int32());
  Tensor truth2 = F::from_list({3, 3, 2, 1, 2}, {1, 5}, dtype::Int64());
  Param p2(0, true);

  Tensor pred2;
  ASSERT_TRUE(oprs->argmxx(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int64>(pred2, truth2, 0.0001f);
}