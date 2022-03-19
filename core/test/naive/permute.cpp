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
using Param = param::permute;

TEST(Naive, Permute) {
  OpNaiveImpl oprs;

  // Group 1
  Tensor inp1 =
      F::from_list({-19, -5, -13, 17,  -19, -19, 8, 4,   -4, -11, -5,  18,
                    4,   -7, -13, -17, 4,   4,   6, -12, -5, 5,   -16, -15},
                   {3, 2, 4}, dtype::Int32());
  Tensor truth1 =
      F::from_list({-19, -4, 4,  -5,  -11, 4, -13, -5,  6,   17, 18,  -12,
                    -19, 4,  -5, -19, -7,  5, 8,   -13, -16, 4,  -17, -15},
                   {2, 4, 3}, dtype::Int32());
  Param p1({1, 2, 0});

  Tensor pred1;
  ASSERT_TRUE(oprs.permute(inp1, pred1, p1).is_ok());
  assert_same_data<nn_int32>(pred1, truth1, 0.0001f);

  // Group 2
  Tensor inp2 = F::from_list(
      {-151, -46,  -9,   -62,  -158, -74,  35,   -10,  -123, -94,  -122, 58,
       -124, 139,  -173, -137, -178, 116,  52,   -92,  -14,  -176, -133, -109,
       -114, -157, -186, 46,   -78,  -144, 155,  -60,  47,   150,  -133, -58,
       -17,  -161, -36,  11,   133,  -170, -149, -155, 10,   -118, -112, -103,
       -110, 183,  29,   21,   189,  -85,  83,   -186, -114, -104, -171, -116,
       -110, 88,   -130, 42,   106,  120,  -94,  -77,  49,   74,   96,   -28},
      {6, 3, 2, 2}, dtype::Int32());
  Tensor truth2 = F::from_list(
      {-151, -124, -114, -17,  -110, -110, -158, -178, -78,  133,  189,  106,
       -123, -14,  47,   10,   -114, 49,   -46,  139,  -157, -161, 183,  88,
       -74,  116,  -144, -170, -85,  120,  -94,  -176, 150,  -118, -104, 74,
       -9,   -173, -186, -36,  29,   -130, 35,   52,   155,  -149, 83,   -94,
       -122, -133, -133, -112, -171, 96,   -62,  -137, 46,   11,   21,   42,
       -10,  -92,  -60,  -155, -186, -77,  58,   -109, -58,  -103, -116, -28},
      {2, 2, 3, 6}, dtype::Int32());
  Param p2({2, 3, 1, 0});

  Tensor pred2;
  ASSERT_TRUE(oprs.permute(inp2, pred2, p2).is_ok());
  assert_same_data<nn_int32>(pred2, truth2, 0.0001f);

  // Group 3
  Tensor inp3 = F::from_list({-144.88911173213074, -177.9092558668978}, {1, 2},
                             dtype::Float64());
  Tensor truth3 = F::from_list({-144.88911173213074, -177.9092558668978},
                               {2, 1}, dtype::Float64());
  Param p3({1, 0});

  Tensor pred3;
  ASSERT_TRUE(oprs.permute(inp3, pred3, p3).is_ok());
  assert_same_data<nn_float64>(pred3, truth3, 0.0001f);
}