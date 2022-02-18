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
  Shape(const std::initializer_list<size_t> init_shape);

  bool is_scalar() const;
  bool is_empty() const;
  bool is_shape(const Shape &rhs) const;
  size_t count() const;

  size_t &operator[](size_t i) { return shape[i]; }
  size_t operator[](size_t i) const { return shape[i]; }

  std::string to_string() const;
};

struct Layout : public Shape {
  enum class Format : uint32_t { Default, NCHW, NHWC };

  DType dtype;
  Format format;

  // ctor
  Layout();
  Layout(const Layout &rhs);
  Layout(const DType &dtype);
  Layout(const Shape &shape, const DType &dtype);
  Layout(const DType &dtype, const Format &format);
  Layout(const Shape &shape, const DType &dtype, const Format &format);

  bool is_layout(const Layout &rhs) const;

  std::string to_string() const;

  size_t content_bytes() const;
};
}  // namespace nncore
