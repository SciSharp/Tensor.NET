#pragma once

#include <iostream>
#include <memory>

#include "core/base/include/layout.h"

namespace nncore {

class RefPtr {
 private:
  std::shared_ptr<void *> m_ref;
  nn_size m_offset;
  bool m_mutable;
  bool m_owned;

 public:
  RefPtr() {
    m_ref = std::make_shared<void *>((void *)nullptr);
    m_offset = 0;
    m_mutable = true;
    m_owned = true;
  }

  RefPtr(void *ref_ptr, const nn_size offset = 0) {
    m_ref = std::make_shared<void *>(ref_ptr);
    m_offset = offset;
    m_mutable = true;
    m_owned = true;
  }

  explicit RefPtr(std::shared_ptr<void *> ref_ptr, const nn_size offset = 0,
                  bool is_mutable = true, bool is_owned = true) {
    m_ref = ref_ptr;
    m_offset = offset;
    m_mutable = is_mutable;
    m_owned = is_owned;
  }

  void *get_ptr() const {
    return static_cast<void *>(
        (*m_ref != NULL) ? static_cast<nn_byte *>(*m_ref) + m_offset : nullptr);
  }

  bool is_mutable() const { return m_mutable; }

  bool is_owned() const { return m_owned; }

  void reset(const void *ptr, nn_size offset = 0, bool is_mutable = true,
             bool is_owner = true);

  RefPtr &operator+=(nn_size offset) {
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
    if (m_owned && m_ref.use_count() == 1) {
      free(*m_ref);
      *m_ref = nullptr;
    }
  }
};

struct Tensor {
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

  Tensor() : m_ref_ptr(RefPtr((void *)nullptr)) {}

  Tensor(const Layout &layout)
      : layout(layout), m_ref_ptr(RefPtr((void *)nullptr)) {
    m_ref_ptr.alloc_memory(layout, layout.dtype);
  }

  Tensor(const Shape &shape, const DType dtype)
      : layout(shape, dtype), m_ref_ptr(RefPtr((void *)nullptr)) {
    m_ref_ptr.alloc_memory(layout, layout.dtype);
  }

  Tensor(const Layout &layout_, void *raw_ptr_)
      : layout(layout_), m_ref_ptr(raw_ptr_) {}

  Tensor(const Layout &layout_, std::shared_ptr<void *> raw_ptr_,
         nn_size offset, bool is_mutable, bool is_owner = false)
      : layout(layout_), m_ref_ptr(raw_ptr_, offset, is_mutable, is_owner) {}

  //! \brief get typed pointer; type check is performed.
  template <typename T>
  T *ptr() const {
    layout.dtype.assert_is_ctype<T>();
    return static_cast<T *>(m_ref_ptr.get_ptr());
  }

  void reset_ptr(void *ptr, nn_size offset, bool is_mutable, bool is_owner);

  bool is_ptr_owner() const { return m_ref_ptr.is_owned(); }

  bool is_mutable() const { return m_ref_ptr.is_mutable(); }

 private:
  RefPtr m_ref_ptr;
};
}  // namespace nncore
