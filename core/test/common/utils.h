#pragma once

#include <iostream>

#include "core/base/include/ndArray.h"
#include "gtest/gtest.h"

namespace nncore {
namespace test {
template <typename ctype>
void assert_same_data(const NDArray &a, const NDArray &b, float err) {
  ASSERT_TRUE(a.layout.dtype.is_ctype<ctype>());
  ASSERT_TRUE(a.layout.is_equivalent_layout(b.layout));
  auto lptr = a.ptr<ctype>();
  auto rptr = b.ptr<ctype>();
  for (size_t i = 0; i < a.layout.count(); i++) {
    // std::cout << "a: " << lptr[i] << " , b: " << rptr[i] << std::endl;
    ASSERT_NEAR(lptr[i], rptr[i], err);
  }
}

template <typename ctype>
void print_data(const NDArray &src) {
  src.layout.dtype.assert_is_ctype<ctype>();
  nn_assert(!src.layout.is_empty(), "Cannot print an empty ndarray.");
  auto ptr = src.ptr<ctype>();
  for (int i = 0; i < src.layout.count(); i++) {
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
