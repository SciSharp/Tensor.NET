#pragma once

#include "core/base/include/layout.h"

namespace nncore {
struct NDArray {
  Layout layout;
  void *raw_ptr;

  NDArray() : raw_ptr(nullptr) {}

  NDArray(const Layout &layout)
      : layout(layout), raw_ptr(alloc_memory(layout, layout.dtype)) {}

  NDArray(const Shape &shape, const DType dtype)
      : layout(shape, dtype), raw_ptr(alloc_memory(shape, dtype)) {}

  NDArray(const Layout &layout_, void *raw_ptr_)
      : layout(layout_), raw_ptr(raw_ptr_) {}

  /*!
   * \brief Get the value by the indcies of dims.
   * This method is not efficient and should not be used in ops.
   */
  template <typename T>
  size_t at(const std::vector<int> &idx) const {
    nn_assert(
        idx.size() == layout.ndim,
        "The count of indices mismatched ndim, the count is %d, and ndim = %d.",
        idx.size(), layout.ndim);
    T r = T(0);
    size_t pos = 0;
    T *dptr = ptr<T>();
    for (int i = 0; i < layout.ndim; i++) {
      pos += idx[i] * layout.stride[i];
    }
    return dptr[pos];
  }

  //! \brief get typed pointer; type check is performed.
  template <typename T>
  T *ptr() const {
    layout.dtype.assert_is_ctype<T>();
    return static_cast<T *>(raw_ptr);
  }

  template <typename T>
  bool equal(const NDArray &rhs, bool with_info = false) const {
    layout.dtype.assert_is_ctype<T>();
    rhs.layout.dtype.assert_is_ctype<T>();
    if (!layout.is_same_layout(rhs.layout)) return false;
    auto lptr = ptr<T>();
    auto rptr = rhs.ptr<T>();
    for (size_t i = 0; i < layout.count(); i++) {
      if (*lptr++ != *rptr++) {
        if (with_info) {
          printf("Different at index %ld, which are %f and %f respectively.", i,
                 static_cast<float>(*(lptr - 1)),
                 static_cast<float>(*(rptr - 1)));
        }
        return false;
      }
    }
    return true;
  }

 private:
  void *alloc_memory(const Shape &shape, const DType &dtype) {
    return malloc(shape.count() * dtype.size());
  }
};
}  // namespace nncore
