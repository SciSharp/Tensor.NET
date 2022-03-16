#pragma once

#include <iostream>

#include "core/base/include/tensor.h"
#include "gtest/gtest.h"

namespace nncore {
namespace test {
template <typename ctype>
void assert_same_data(const Tensor &a, const Tensor &b, float err = 0.000001f) {
  ASSERT_TRUE(a.layout.dtype.is_ctype<ctype>());
  ASSERT_TRUE(a.layout.is_equivalent_layout(b.layout));
  auto lptr = a.ptr<ctype>();
  auto rptr = b.ptr<ctype>();
  for (size_t i = 0; i < a.layout.total_elems(); i++) {
    // std::cout << "a: " << lptr[i] << " , b: " << rptr[i] << std::endl;
    ASSERT_NEAR(lptr[i], rptr[i], err);
  }
}

template <typename ctype>
void assert_same_view(const Tensor &a, const Tensor &b, float err = 0.000001f) {
  ASSERT_TRUE(a.layout.dtype.is_ctype<ctype>());
  ASSERT_TRUE(a.layout.is_equivalent_layout(b.layout));
  ASSERT_TRUE(!a.layout.is_empty());
  auto lptr = a.ptr<ctype>();
  auto rptr = b.ptr<ctype>();
  for (int i = 0; i < (a.layout.ndim >= 4 ? a.layout[3] : 1); i++) {
    for (int j = 0; j < (a.layout.ndim >= 3 ? a.layout[2] : 1); j++) {
      for (int k = 0; k < (a.layout.ndim >= 2 ? a.layout[1] : 1); k++) {
        for (int p = 0; p < a.layout[0]; p++) {
          ctype pred = lptr[i * a.layout.stride[3] + j * a.layout.stride[2] +
                            k * a.layout.stride[1] + p * a.layout.stride[0]];
          ctype truth = rptr[i * b.layout.stride[3] + j * b.layout.stride[2] +
                             k * b.layout.stride[1] + p * b.layout.stride[0]];
          ASSERT_NEAR(pred, truth, err);
        }
      }
    }
  }
}

template <typename ctype>
void print_data(const Tensor &src) {
  src.layout.dtype.assert_is_ctype<ctype>();
  nn_assert(!src.layout.is_empty(), "Cannot print an empty ndarray.");
  auto ptr = src.ptr<ctype>();
  for (int i = 0; i < src.layout.total_elems(); i++) {
    int mod = 1;
    for (int j = 0; j < src.layout.ndim; j++) {
      mod *= src.layout.shape[j];
      if (i % mod == 0) {
        std::cout << "[";
      } else {
        break;
      }
    }
    std::cout << " ";

    std::cout << ptr[i];
    if ((i + 1) % src.layout.shape[0] != 0) std::cout << ",";

    std::cout << " ";
    mod = 1;
    int hit_times = 0;
    for (int j = 0; j < src.layout.ndim; j++) {
      mod *= src.layout.shape[j];
      if ((i + 1) % mod == 0) {
        std::cout << "]";
        hit_times++;
      } else {
        break;
      }
    }
    if (hit_times > 0 && hit_times < src.layout.ndim) {
      std::cout << "," << std::endl;
      for (int j = 0; j < hit_times; j++) {
        std::cout << " ";
      }
    }
  }
  std::cout << std::endl;
}
}  // namespace test

}  // namespace nncore
