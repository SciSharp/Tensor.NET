#include "core/op/naive/ops.h"
#include "core/test/common/factory.h"
#include "core/test/common/utils.h"
#include "gtest/gtest.h"

using namespace nncore;
using namespace test;
using namespace opr;
using namespace opr::naive;

using F = NDArrayFactory;
using Param = param::linspace;

TEST(Naive, Linspace) {
  OpBase *oprs = OpNaiveImpl::get_instance();

  // Group 1
  Tensor t1({50}, dtype::Float64());
  Tensor truth1 = F::from_list(
      {.0,         0.20408163, 0.40816327, 0.6122449,  0.81632653, 1.02040816,
       1.2244898,  1.42857143, 1.63265306, 1.83673469, 2.04081633, 2.24489796,
       2.44897959, 2.65306122, 2.85714286, 3.06122449, 3.26530612, 3.46938776,
       3.67346939, 3.87755102, 4.08163265, 4.28571429, 4.48979592, 4.69387755,
       4.89795918, 5.10204082, 5.30612245, 5.51020408, 5.71428571, 5.91836735,
       6.12244898, 6.32653061, 6.53061224, 6.73469388, 6.93877551, 7.14285714,
       7.34693878, 7.55102041, 7.75510204, 7.95918367, 8.16326531, 8.36734694,
       8.57142857, 8.7755102,  8.97959184, 9.18367347, 9.3877551,  9.59183673,
       9.79591837, 10.0},
      {50}, dtype::Float64());
  Param p1(0, 10, 50, true);

  ASSERT_TRUE(oprs->linspace(t1, p1).is_ok());
  assert_same_data<nn_float64>(t1, truth1, 0.0001f);

  // Group 2
  Tensor t2({20}, dtype::Float64());
  Tensor truth2 =
      F::from_list({-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3,
                    .0,   0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1,  2.4,  2.7},
                   {20}, dtype::Float64());
  Param p2(-3, 3, 20, false);

  ASSERT_TRUE(oprs->linspace(t2, p2).is_ok());
  assert_same_data<nn_float64>(t2, truth2, 0.0001f);
}