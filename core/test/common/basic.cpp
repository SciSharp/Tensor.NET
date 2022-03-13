#include "core/base/include/ndArray.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;

using F = NDArrayFactory;

TEST(Basic, reshape) {
  // group1
  auto inp = F::from_list({1, 2, 3, 4, 5, 6}, {2, 3}, dtype::Int32());
  auto truth = F::from_list({1, 2, 3, 4, 5, 6}, {3, 2}, dtype::Int32());
  Layout pred_layout;
  ASSERT_TRUE(inp.layout.try_reshape(pred_layout, {3, 2}));
  inp.layout = pred_layout;
  assert_same_view<int>(inp, truth);

  // group2
  auto inp2 = F::from_list({-11, 13, -20, 3,  14, 18, 6,  10, 1, 19, 0,   -13,
                            -1,  -4, -9,  -4, 16, -9, -2, -5, 5, -7, -10, -17},
                           {2, 3, 4}, dtype::Int32());
  auto truth2 =
      F::from_list({-11, 13, -20, 3,  14, 18, 6,  10, 1, 19, 0,   -13,
                    -1,  -4, -9,  -4, 16, -9, -2, -5, 5, -7, -10, -17},
                   {4, 6}, dtype::Int32());
  Layout pred_layout2;
  ASSERT_TRUE(inp2.layout.try_reshape(pred_layout2, {2, 2, 3, 2}));
  ASSERT_TRUE(pred_layout2.try_reshape(pred_layout2, {4, 6}));
  inp2.layout = pred_layout2;
  assert_same_view<int>(inp2, truth2);

  // group3
  auto inp3 = F::from_list({-11, 13, -20, 3,  14, 18, 6,  10, 1, 19, 0,   -13,
                            -1,  -4, -9,  -4, 16, -9, -2, -5, 5, -7, -10, -17},
                           {24}, dtype::Int32());
  auto truth3 =
      F::from_list({-11, 13, -20, 3,  14, 18, 6,  10, 1, 19, 0,   -13,
                    -1,  -4, -9,  -4, 16, -9, -2, -5, 5, -7, -10, -17},
                   {2, 2, 3, 2}, dtype::Int32());
  Layout pred_layout3;
  ASSERT_TRUE(inp3.layout.try_reshape(pred_layout3, {6, 2, 1, 2}));
  ASSERT_TRUE(pred_layout3.try_reshape(pred_layout3, {2, 2, 3, 2}));
  inp3.layout = pred_layout3;
  assert_same_view<int>(inp3, truth3);
}