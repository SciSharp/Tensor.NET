#pragma once

#include <memory>

#include "core/base/include/layout.h"

namespace nncore {

class RefPtr {
  std::shared_ptr<void *> m_ref;
  size_t m_offset;
  bool m_mutable;

 public:
  RefPtr() {
    m_ref = std::make_shared<void *>((void *)nullptr);
    m_offset = 0;
    m_mutable = true;
  }

  RefPtr(void *ref_ptr, const size_t offset = 0) {
    m_ref = std::make_shared<void *>(ref_ptr);
    m_offset = offset;
    m_mutable = true;
  }

  explicit RefPtr(std::shared_ptr<void *> ref_ptr, const size_t offset = 0,
                  bool is_mutable = true) {
    m_ref = ref_ptr;
    m_offset = offset;
    m_mutable = is_mutable;
  }

  void *get_ptr() const {
    return static_cast<void *>(
        (*m_ref != NULL) ? static_cast<nn_byte *>(*m_ref) + m_offset : nullptr);
  }

  bool is_mutable() const { return m_mutable; }

  void reset(const void *ptr, size_t offset = 0);

  RefPtr &operator+=(size_t offset) {
    m_offset += offset;
    return *this;
  }

  bool operator==(const RefPtr &other) const {
    return *m_ref == *other.m_ref && m_offset == other.m_offset;
  }

  template <typename T>
  T *ptr() const {
    return static_cast<T *>(get_ptr());
  }

  void alloc_memory(const Shape &shape, const DType &dtype) {
    *m_ref = malloc(shape.total_elems() * dtype.size());
  }

  ~RefPtr() {
    free(*m_ref);
    *m_ref = nullptr;
  }
};

struct NDArray {
  Layout layout;

  /*
   * \brief This method should be always used on an empty array!
   * It is mainly used to delay the specification of array. Take
   * care of using it!
   */
  void relayout(const Layout &layout) {
    this->layout = layout;
    m_ref_ptr.alloc_memory(layout, layout.dtype);
  }

  NDArray() : m_ref_ptr(RefPtr((void *)nullptr)) {}

  NDArray(const Layout &layout)
      : layout(layout), m_ref_ptr(RefPtr((void *)nullptr)) {
    m_ref_ptr.alloc_memory(layout, layout.dtype);
  }

  NDArray(const Shape &shape, const DType dtype)
      : layout(shape, dtype), m_ref_ptr(RefPtr((void *)nullptr)) {
    m_ref_ptr.alloc_memory(layout, layout.dtype);
  }

  NDArray(const Layout &layout_, void *raw_ptr_)
      : layout(layout_), m_ref_ptr(raw_ptr_) {}

  //! \brief get typed pointer; type check is performed.
  template <typename T>
  T *ptr() const {
    layout.dtype.assert_is_ctype<T>();
    return static_cast<T *>(m_ref_ptr.get_ptr());
  }

  void reset_ptr(void *ptr, size_t offset);

 private:
  RefPtr m_ref_ptr;
};
}  // namespace nncore
