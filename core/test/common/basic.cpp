#include "core/op/naive/ops.h"
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

  // group4 combined with matmul
  Tensor a4 = F::from_list(
      {17.466985062517146,  6.7664066410803585,  14.843100731907732,
       -4.6390678878805645, -11.721863661802509, -10.874290362288598,
       -17.450436421583543, -6.755345064323057,  8.327839373672429,
       5.832145729343814,   10.187632145963143,  -6.933134973938785,
       5.4345048380072605,  18.830019942772367,  -0.08915236388483905,
       -13.344241127102215, 19.56888266289109,   -2.325354778121529,
       -18.35114478739234,  16.080997940178044,  19.130843119578557,
       -7.780265244261599,  7.957347818762777,   -4.786855939893734},
      {2, 4, 3}, dtype::Float64());
  Tensor b4 =
      F::from_list({0.5741403351746364, -2.052231433802003, 25.24007024109701,
                    29.858219492503345, -18.67625475563329, 12.918972076270109,
                    -1.753392156743466, 23.42291952066134, 26.036713810702366,
                    -7.365878774868115, 20.185043802968643, -12.90536007406542},
                   {2, 6}, dtype::Float64());
  Tensor truth4 = F::from_list(
      {220.2063906086251, 91.81153535065636, 974.6173556521721,
       -251.06043201452295, -317.95010976837546, -803.5171742210955,
       212.12455323560656, -27.334838431359515, 640.2676444441263,
       663.7994184691494, -634.2694114728648, 550.3227877495598,
       -144.46959658080647, -101.97269754089359, -221.45579419121552,
       -200.01402938009352, 195.8067188946301, 651.3100820493304},
      {3, 2, 3}, dtype::Float64());
  Layout pred_layout_a4;
  ASSERT_TRUE(a4.layout.try_reshape(pred_layout_a4, {3, 2, 4}));
  a4.layout = pred_layout_a4;
  Layout pred_layout_b4;
  ASSERT_TRUE(b4.layout.try_reshape(pred_layout_b4, {4, 3}));
  b4.layout = pred_layout_b4;
  Tensor pred4;
  using Param = param::matmul;
  Param p4;
  opr::OpBase* oprs = opr::naive::OpNaiveImpl::get_instance();
  ASSERT_TRUE(oprs->matmul(a4, b4, pred4, p4).is_ok());

  assert_same_data<double>(pred4, truth4, 0.000001);
}