#pragma once

#include "core/base/include/layout.h"

namespace nncore {
struct NDArray {
  void *raw_ptr;
  Layout layout;

  NDArray() : raw_ptr(nullptr) {}

  NDArray(void *raw_ptr_, const Layout &layout_)
      : raw_ptr(raw_ptr_), layout(layout_) {}

  /*!
   * \brief Get the absolute index by the indcies of dims.
   * This method is not efficient and should not be used in ops.
   */
  template <typename T>
  size_t at(std::initializer_list<int> idx) {
    size_t n = idx.size();
    nn_assert(
        n > 0 && n <= layout.ndim,
        "The count of indices is out of range, the count is %d, and ndim = %d.",
        n, layout.ndim);
    T r = 0;
    auto iptr = idx.begin();
    size_t pos = *iptr;
    T *dptr = ptr<T>();
    int i = 1;
    for (; iptr != idx.end(); iptr++) {
      pos = pos * layout[i + 1] + *iptr;
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
};
}  // namespace nncore
