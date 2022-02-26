#include "core/base/include/layout.h"

#include <cstring>
#include <string>

namespace nncore {
Shape::Shape(const std::vector<size_t> &init_shape) {
  nn_assert(init_shape.size() < MAX_NDIM,
            "The shape you specified has too many dims, which is %lu, the "
            "max is %lu\n",
            init_shape.size(), MAX_NDIM);
  ndim = init_shape.size();
  for (int i = 0; i < ndim; i++) {
    shape[i] = init_shape[ndim - 1 - i];
  }
}

Shape::Shape(const std::initializer_list<size_t> init_shape)
    : Shape(std::vector<size_t>{init_shape}) {}

bool Shape::is_scalar() const { return ndim == 1 && shape[0] == 1; }

bool Shape::is_empty() const {
  if (ndim == 0) return true;
  for (size_t i = 0; i < ndim; i++) {
    if (shape[i] == 0) return true;
  }
  return false;
}

bool Shape::is_shape(const Shape &rhs) const {
  if (ndim != rhs.ndim) return false;
  for (size_t i = 0; i < ndim; i++) {
    if (shape[i] != rhs.shape[i]) return false;
  }
  return true;
}

size_t Shape::count() const {
  size_t r = 1;
  for (size_t i = 0; i < ndim; i++) r *= shape[i];
  return r;
}

std::string Shape::to_string() const {
  std::string r = "{";
  if (ndim > 0) {
    for (int i = 0; i < ndim; i++) {
      r += std::to_string(shape[i]);
      if (i != ndim - 1) r += ", ";
    }
  }
  r += "}";
  return r;
}

Layout::Layout() : dtype(), format(Format::Default) {}

Layout::Layout(const Layout &rhs) : dtype(rhs.dtype), format(rhs.format) {}

Layout::Layout(const DType &dtype) : dtype(dtype), format(Format::Default) {}

Layout::Layout(const Shape &shape, const DType &dtype)
    : Shape(shape), dtype(dtype), format(Format::Default) {}

Layout::Layout(const DType &dtype, const Format &format)
    : dtype(dtype), format(format) {}

Layout::Layout(const Shape &shape, const DType &dtype, const Format &format)
    : Shape(shape), dtype(dtype), format(format) {}

bool Layout::is_same_layout(const Layout &rhs) const {
  return dtype == rhs.dtype && format == rhs.format && is_shape(rhs);
}

std::string Layout::to_string() const {
  std::string r = "({";
  if (ndim > 0) {
    for (int i = 0; i < ndim; i++) {
      r += std::to_string(shape[i]);
      if (i != ndim - 1) r += ", ";
    }
  }
  r += "}, dtype = ";
  r += dtype.name();
  r += ")";
  return r;
}

size_t Layout::content_bytes() const { return count() * dtype.size(); }

}  // namespace nncore
