#include "core/base/include/layout.h"

#include <cstring>
#include <numeric>
#include <string>

#define rep(i, n) for (auto i = decltype(n){0}; i < (n); ++i)

namespace nncore {
Shape::Shape(const std::vector<size_t> &init_shape) {
  nn_assert(init_shape.size() <= MAX_NDIM,
            "The shape you specified has too many dims, which is %lu, the "
            "max is %d\n",
            init_shape.size(), MAX_NDIM);
  ndim = init_shape.size();
  for (int i = 0; i < ndim; i++) {
    shape[i] = init_shape[ndim - 1 - i];
  }
  for (int i = ndim; i < MAX_NDIM; i++) {
    shape[i] = 1;
  }
}

Shape::Shape(size_t *init_shape, int ndim) {
  this->ndim = ndim;
  memcpy(shape, init_shape, ndim * sizeof(size_t));
}

Shape::Shape(const std::initializer_list<size_t> &init_shape)
    : Shape(std::vector<size_t>{init_shape}) {}

bool Shape::is_scalar() const { return ndim == 1 && shape[0] == 1; }

bool Shape::is_empty() const {
  if (ndim == 0) return true;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] == 0) return true;
  }
  return false;
}

bool Shape::is_shape(const Shape &rhs) const {
  if (ndim != rhs.ndim) return false;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] != rhs.shape[i]) return false;
  }
  return true;
}

bool Shape::is_equivalent_shape(const Shape &rhs) const {
  int min_ndim = ndim > rhs.ndim ? rhs.ndim : ndim;
  for (int i = 0; i < min_ndim; i++) {
    if (shape[i] != rhs.shape[i]) return false;
  }
  for (int i = min_ndim; i < ndim; i++) {
    if (shape[i] != 1) return false;
  }
  for (int i = min_ndim; i < rhs.ndim; i++) {
    if (rhs.shape[i] != 1) return false;
  }
  return true;
}

size_t Shape::total_elems() const {
  size_t r = 1;
  for (int i = 0; i < ndim; i++) r *= shape[i];
  return r;
}

std::string Shape::to_string() const {
  std::string r = "{";
  if (ndim > 0) {
    for (int i = 0; i < ndim; i++) {
      r += std::to_string(shape[ndim - 1 - i]);
      if (i != ndim - 1) r += ", ";
    }
  }
  r += "}";
  return r;
}

Layout::Layout() : dtype() {}

Layout::Layout(const DType &dtype) : dtype(dtype) {}

Layout::Layout(const Shape &shape, const DType &dtype)
    : Shape(shape), dtype(dtype) {
  init_contiguous_stride();
}

Layout::Layout(const Shape &shape, const std::vector<size_t> &stride,
               const DType &dtype)
    : Shape(shape), dtype(dtype) {
  nn_assert(shape.ndim == stride.size(),
            "Size of shape mismatched that of stride.");
  for (int i = 0; i < shape.ndim; i++) this->stride[i] = stride[ndim - i - 1];
}

size_t Layout::init_contiguous_stride() {
  nn_assert(ndim);
  nn_assert(ndim <= Layout::MAX_NDIM);
  size_t s = 1;
  for (int i = 0; i < ndim; i++) {
    stride[i] = s;
    s *= shape[i];
  }
  return s;
}

size_t Layout::init_contiguous_stride(const Shape &shape) {
  this->Shape::operator=(shape);
  return init_contiguous_stride();
}

void Layout::broadcast_inplace(const Shape &target) {
  nn_assert(ndim && target.ndim, "Empty tensor in broadcast.");

  if (is_scalar()) {
    ndim = target.ndim;
    for (int i = 0; i < target.ndim; i++) {
      shape[i] = target.shape[i];
      stride[i] = (target.shape[i] == 1);
    }
    return;
  }

  nn_assert(target.ndim >= ndim,
            "dimension for broadcast less than "
            "dst_shape: src_shape=%s dst_shape=%s",
            to_string().c_str(), target.to_string().c_str());
  for (int i = 0; i < target.ndim; ++i) {
    size_t cur_shape = (i < ndim ? shape[i] : 1),
           cur_stride = (i < ndim ? stride[i] : 0);
    if (target.shape[i] != cur_shape) {
      nn_assert(cur_shape == 1 || cur_stride == 0,
                "broadcast on dim with shape not equal to 1: "
                "src_shape=%s dst_shape=%s",
                to_string().c_str(), target.to_string().c_str());
      shape[i] = target.shape[i];
      stride[i] = 0;
    } else {
      shape[i] = cur_shape;
      stride[i] = cur_stride;
    }
  }
  ndim = target.ndim;
}

Layout Layout::broadcast(const Shape &target) const {
  Layout result(dtype);

  result.broadcast_inplace(target);
  return result;
}

bool Layout::is_same_layout(const Layout &rhs) const {
  if (dtype != rhs.dtype || !is_shape(rhs)) return false;
  for (int i = 0; i < ndim; i++) {
    if (stride[i] != rhs.stride[i]) return false;
  }
  return true;
}

bool Layout::is_equivalent_layout(const Layout &rhs) const {
  return dtype == rhs.dtype && is_equivalent_shape(rhs);
}

void Layout::offset_to_indices(size_t offset, size_t *indices) const {
  for (int i = 0; i < ndim; i++) {
    int idx = ndim - i - 1;
    indices[idx] = offset / stride[idx];
    offset %= stride[idx];
  }
}

size_t Layout::indices_to_offset(size_t *indices) const {
  size_t res = 0;
  for (int i = 0; i < ndim; i++) {
    res += indices[i] * stride[i];
  }
  return res;
}

std::string Layout::to_string() const {
  std::string r = "(";
  if (!ndim) {
    r += " Scalar";
  } else {
    r += "shape = {";
    for (int i = 0; i < ndim; i++) {
      r += std::to_string(shape[ndim - 1 - i]);
      if (i != ndim - 1) r += ", ";
    }
    r += "}, stride = {";
    for (int i = 0; i < ndim; i++) {
      r += std::to_string(stride[ndim - 1 - i]);
      if (i != ndim - 1) r += ", ";
    }
    r += "}";
  }
  r += ", dtype = ";
  r += dtype.name();
  r += ")";
  return r;
}

size_t Layout::content_bytes() const { return total_elems() * dtype.size(); }

Layout Layout::dimshuffle(const std::vector<size_t> &dims) const {
  Layout res{dtype};
  res.ndim = this->ndim;
  nn_assert(dims.size() == this->ndim);
  auto ndim = this->ndim;
  rep(i, ndim) {
    auto dest = dims[res.ndim - i - 1];
    nn_assert(dest < ndim);
    res.shape[i] = this->shape[dest];
    res.stride[i] = this->stride[dest];
  }
  return res;
}

Layout Layout::remove_axis(size_t idx) const {
  Layout res{*this};
  res.remove_axis_inplace(idx);
  return res;
}

void Layout::remove_axis_inplace(size_t axis) {
  nn_assert(ndim >= 2 && axis < ndim);
  --ndim;
  for (int i = axis; i < ndim; ++i) {
    shape[i] = shape[i + 1];
    stride[i] = stride[i + 1];
  }
}

void Layout::add_axis_inplace(size_t axis, size_t shape, size_t stride) {
  nn_assert(ndim + 1 <= MAX_NDIM && axis <= ndim && shape,
            "can not add axis at %zu (current ndim %d, MAX_NDIM %d)", axis,
            ndim, MAX_NDIM);
  ndim++;
  for (int i = ndim - 1; i > axis; i--) {
    this->shape[i] = this->shape[i - 1];
    this->stride[i] = this->stride[i - 1];
  }
  this->shape[axis] = shape;
  this->stride[axis] = stride;
}

bool Layout::is_contiguous() const {
  size_t expected = 1;
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] != 1 && stride[i] != expected) return false;
    expected *= shape[i];
  }
  // empty tensors are not contiguous
  return expected != 0;
}

Layout Layout::collapse_contiguous() const {
  // assert_valid(layout);
  nn_assert(ndim);
  Layout res{*this};

  // remove all dims with shape 1
  for (int i = 0; i <= res.ndim - 1 && res.ndim >= 2; ++i) {
    if (!res.shape[i]) {
      // empty tensor
      res.ndim = 1;
      res.shape[0] = 0;
      res.stride[0] = 1;
      return res;
    }
    if (res.shape[i] == 1) res.remove_axis_inplace(i);
  }

  if (res.ndim == 1) {
    if (res.shape[0] <= 1) {
      // make it the "most canonical" contiguous layout for scalars or
      // empty tensors
      res.stride[0] = 1;
    }
    return res;
  }

  nn_assert(res.ndim && res.shape[res.ndim - 1]);
  for (size_t i = 1; i <= res.ndim - 1; ++i) {
    nn_assert(res.shape[i]);
    if (res.stride[i] == res.stride[i - 1] * res.shape[i - 1]) {
      res.shape[i] *= res.shape[i - 1];
      res.stride[i] = res.stride[i - 1];
      res.remove_axis_inplace(i - 1);
    }
  }
  return res;
}

bool Layout::try_reshape(Layout &result, const Shape &tshp,
                         bool is_image) const {
  nn_assert(tshp.ndim);

  bool is_empty_shape = false;
  for (int i = 0; i < tshp.ndim; ++i) {
    if (!tshp.shape[i]) {
      is_empty_shape = true;
      break;
    }
  }

  nn_assert(tshp.ndim && total_elems() == tshp.total_elems(),
            "number of elements do not match "
            "in reshape: src=%s dest=%s",
            static_cast<const Shape &>(*this).to_string().c_str(),
            tshp.to_string().c_str());

  // So far only the swap of width and height is supported.
  if (is_image) {
    nn_assert(this->ndim >= 2 && tshp.ndim >= 2 &&
              this->shape[0] == tshp.shape[1] &&
              this->shape[1] == tshp.shape[0]);
    for (int i = 2; i < this->ndim; i++) {
      if (tshp.ndim > i && tshp.shape[i] != this->shape[i]) {
        return false;
      }
    }
    result.dtype = this->dtype;
    result.Shape::operator=(tshp);
    std::swap(result.stride[0], result.stride[1]);
    return true;
  }

  auto cont = collapse_contiguous();
  result.dtype = this->dtype;
  result.Shape::operator=(tshp);

  if (is_empty_shape) {
    result.init_contiguous_stride();
    return true;
  }

  // size_t sdim = 0, prod = 1, cont_sdim = 0;
  // for (size_t i = 0; i < tshp.ndim; ++i) {
  //   nn_assert(cont_sdim < cont.ndim);
  //   prod *= result.shape[i];
  //   if (prod > cont.shape[cont_sdim]) return false;

  //   if (prod == cont.shape[cont_sdim] &&
  //       (i + 1 >= tshp.ndim || tshp.shape[i + 1] != 1)) {
  //     auto s = cont.stride[cont_sdim];
  //     for (int j = i; j >= static_cast<int>(sdim); --j) {
  //       result.stride[j] = s;
  //       s *= result.shape[j];
  //     }
  //     ++cont_sdim;
  //     sdim = i + 1;
  //     prod = 1;
  //   }
  // }
  // nn_assert(cont_sdim == cont.ndim);

  result.init_contiguous_stride();

  return true;
}

Layout Layout::reshape(const Shape &shape, bool is_image) const {
  Layout ret;
  auto succ = try_reshape(ret, shape, is_image);
  nn_assert(succ, "can not reshape from %s to %s", to_string().c_str(),
            shape.to_string().c_str());
  return ret;
}

}  // namespace nncore
