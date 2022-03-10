#pragma once

#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "core/base/include/dtype.h"
#include "core/macro.h"

namespace nncore {
struct Shape {
  static constexpr size_t MAX_NDIM = NN_MAX_NDIM;

  size_t shape[MAX_NDIM];
  size_t ndim = 0;

  // ctor
  Shape() = default;
  Shape(const Shape &rhs) = default;
  Shape(const std::vector<size_t> &init_shape);
  Shape(const std::initializer_list<size_t> &init_shape);

  bool is_scalar() const;
  bool is_empty() const;
  bool is_shape(const Shape &rhs) const;

  /*
   * \brief Return if the two shapes are equivalent, which means that one could
   * directly reshape to the other without changing the arrangement of elements.
   *
   * For instance, {1, 1, 2, 3} and {1, 2, 3} are equivalent. Instead, {1, 2, 3}
   * and {3, 2, 1} are not, though {1, 2, 3} could be reshaped to {3, 2, 1}.
   */
  bool is_equivalent_shape(const Shape &rhs) const;
  size_t count() const;

  size_t &operator[](size_t i) { return shape[i]; }
  size_t operator[](size_t i) const { return shape[i]; }

  std::string to_string() const;
};

struct Layout : public Shape {
  enum class Format : uint32_t { Default, NCHW, NHWC };

  DType dtype;
  Format format;
  size_t stride[MAX_NDIM];

  // ctor
  Layout();
  Layout(const Layout &rhs) = default;
  Layout(const DType &dtype);
  Layout(const Shape &shape, const DType &dtype);
  Layout(const DType &dtype, const Format &format);
  Layout(const Shape &shape, const DType &dtype, const Format &format);
  Layout(const Shape &shape, const std::vector<size_t> &stride,
         const DType &dtype);
  Layout(const Shape &shape, const std::vector<size_t> &stride,
         const DType &dtype, const Format &format);

  /*
   * \brief Automatically fill the stride of this layout with its current shape.
   * This also means that no broadcast is on this layout.
   * \return The current layout itself.
   */
  const Layout &auto_stride();

  void self_broadcast(const Shape &target);
  Layout broadcast(const Shape &target) const;

  bool is_same_layout(const Layout &rhs) const;

  /*
   * \brief Return if the two layouts are equivalent, which means that one could
   * directly reshape to the other without changing the arrangement of elements.
   *
   * For instance, {1, 1, 2, 3} and {1, 2, 3} are equivalent. Instead, {1, 2, 3}
   * and {3, 2, 1} are not, though {1, 2, 3} could be reshaped to {3, 2, 1}.
   */
  bool is_equivalent_layout(const Layout &rhs) const;

  std::string to_string() const;

  size_t content_bytes() const;
};
}  // namespace nncore
