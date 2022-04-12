#include "core/base/include/layout.h"

#include <cstring>
#include <numeric>
#include <string>

#define rep(i, n) for (auto i = decltype(n){0}; i < (n); ++i)

namespace nncore {
Shape::Shape(const std::vector<nn_size> &init_shape) {
  nn_assert(init_shape.size() <= MAX_NDIM,
            "The shape you specified has too many dims, which is %lu, the "
            "max is %d\n",
            init_shape.size(), MAX_NDIM);
  ndim = init_shape.size();
  memcpy(shape, init_shape.data(), sizeof(nn_size) * ndim);
  // for (nn_size i = ndim; i < MAX_NDIM; i++) {
  //   shape[i] = 1;
  // }
}

Shape::Shape(nn_size *init_shape, nn_size ndim) {
  this->ndim = ndim;
  memcpy(shape, init_shape, ndim * sizeof(nn_size));
}

Shape::Shape(const std::initializer_list<nn_size> &init_shape)
    : Shape(std::vector<nn_size>{init_shape}) {}

bool Shape::is_scalar() const { return ndim == 1 && shape[0] == 1; }

bool Shape::is_empty() const {
  if (ndim == 0) return true;
  for (nn_size i = 0; i < ndim; i++) {
    if (shape[i] == 0) return true;
  }
  return false;
}

bool Shape::is_shape(const Shape &rhs) const {
  if (ndim != rhs.ndim) return false;
  for (nn_size i = 0; i < ndim; i++) {
    if (shape[i] != rhs.shape[i]) return false;
  }
  return true;
}

bool Shape::is_equivalent_shape(const Shape &rhs) const {
  nn_size min_ndim = ndim > rhs.ndim ? rhs.ndim : ndim;
  for (nn_size i = 0; i < min_ndim; i++) {
    if (shape[ndim - i - 1] != rhs.shape[rhs.ndim - i - 1]) return false;
  }
  for (nn_size i = min_ndim; i < ndim; i++) {
    if (shape[i] != 1) return false;
  }
  for (nn_size i = min_ndim; i < rhs.ndim; i++) {
    if (rhs.shape[i] != 1) return false;
  }
  return true;
}

nn_size Shape::total_elems() const {
  nn_size r = 1;
  for (nn_size i = 0; i < ndim; i++) r *= shape[i];
  return r;
}

std::string Shape::to_string() const {
  std::string r = "{";
  if (ndim > 0) {
    for (nn_size i = 0; i < ndim; i++) {
      r += std::to_string(shape[i]);
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

Layout::Layout(const Shape &shape, const std::vector<nn_size> &stride,
               const DType &dtype)
    : Shape(shape), dtype(dtype) {
  nn_assert(shape.ndim == stride.size(),
            "Size of shape mismatched that of stride.");
  for (nn_size i = 0; i < shape.ndim; i++) this->stride[i] = stride[i];
}

nn_size Layout::init_contiguous_stride() {
  nn_assert(ndim);
  nn_assert(ndim <= Layout::MAX_NDIM);
  nn_size s = 1;
  for (nn_size i = 0; i < ndim; i++) {
    stride[ndim - i - 1] = s;
    s *= shape[ndim - i - 1];
  }
  return s;
}

nn_size Layout::init_contiguous_stride(const Shape &shape) {
  this->Shape::operator=(shape);
  return init_contiguous_stride();
}

void Layout::broadcast_inplace(const Shape &target) {
  nn_assert(ndim && target.ndim, "Empty tensor in broadcast.");

  if (is_scalar()) {
    ndim = target.ndim;
    for (nn_size i = 0; i < target.ndim; i++) {
      shape[i] = target.shape[i];
      stride[i] = (target.shape[i] == 1);
    }
    return;
  }

  nn_assert(target.ndim >= ndim,
            "dimension for broadcast less than "
            "dst_shape: src_shape=%s dst_shape=%s",
            to_string().c_str(), target.to_string().c_str());
  for (nn_size i = 0; i < target.ndim; ++i) {
    nn_size target_idx = target.ndim - i - 1;
    nn_size cur_shape = i < ndim ? shape[ndim - i - 1] : 1,
            cur_stride = i < ndim ? stride[ndim - i - 1] : 0;
    if (target.shape[target_idx] != cur_shape) {
      nn_assert(cur_shape == 1 || cur_stride == 0,
                "broadcast on dim with shape not equal to 1: "
                "src_shape=%s dst_shape=%s",
                to_string().c_str(), target.to_string().c_str());
      shape[target_idx] = target.shape[target_idx];
      stride[target_idx] = 0;
    } else {
      shape[target_idx] = cur_shape;
      stride[target_idx] = cur_stride;
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
  for (nn_size i = 0; i < ndim; i++) {
    if (stride[i] != rhs.stride[i]) return false;
  }
  return true;
}

bool Layout::is_equivalent_layout(const Layout &rhs) const {
  return dtype == rhs.dtype && is_equivalent_shape(rhs);
}

void Layout::offset_to_indices(nn_size offset, nn_size *indices) const {
  for (nn_size i = 0; i < ndim; i++) {
    int mod = stride[i] != 0 ? stride[i] : shape[i];
    indices[i] = offset / mod;
    offset %= mod;
  }
}

nn_size Layout::indices_to_offset(nn_size *indices) const {
  nn_size res = 0;
  for (nn_size i = 0; i < ndim; i++) {
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
    for (nn_size i = 0; i < ndim; i++) {
      r += std::to_string(shape[i]);
      if (i != ndim - 1) r += ", ";
    }
    r += "}, stride = {";
    for (nn_size i = 0; i < ndim; i++) {
      r += std::to_string(stride[i]);
      if (i != ndim - 1) r += ", ";
    }
    r += "}";
  }
  r += ", dtype = ";
  r += dtype.name();
  r += ")";
  return r;
}

nn_size Layout::content_bytes() const { return total_elems() * dtype.size(); }

Layout Layout::dimshuffle(const std::vector<nn_size> &dims) const {
  Layout res{dtype};
  res.ndim = this->ndim;
  nn_assert(dims.size() == this->ndim);
  auto ndim = this->ndim;
  rep(i, ndim) {
    auto dest = dims[i];
    nn_assert(dest < ndim);
    res.shape[i] = this->shape[dest];
    res.stride[i] = this->stride[dest];
  }
  return res;
}

Layout Layout::remove_axis(nn_size idx) const {
  Layout res{*this};
  res.remove_axis_inplace(idx);
  return res;
}

void Layout::remove_axis_inplace(nn_size axis) {
  nn_assert(ndim >= 2 && axis < ndim);
  ndim--;
  for (nn_size i = axis; i < ndim; i++) {
    shape[i] = shape[i + 1];
    stride[i] = stride[i + 1];
  }
}

void Layout::add_axis_inplace(nn_size axis, nn_size shape, nn_size stride) {
  nn_assert(ndim + 1 <= MAX_NDIM && axis <= ndim && shape,
            "can not add axis at %d (current ndim %d, MAX_NDIM %d)", axis, ndim,
            MAX_NDIM);
  ndim++;
  for (int i = ndim - 1; i > axis; i--) {
    this->shape[i] = this->shape[i - 1];
    this->stride[i] = this->stride[i - 1];
  }
  this->shape[axis] = shape;
  this->stride[axis] = stride;
}

bool Layout::is_contiguous() const {
  nn_size expected = 1;
  for (nn_size i = 0; i < ndim; ++i) {
    if (shape[ndim - i - 1] != 1 && stride[ndim - i - 1] != expected)
      return false;
    expected *= shape[ndim - i - 1];
  }
  // empty tensors are not contiguous
  return expected != 0;
}

Layout Layout::collapse_contiguous() const {
  // assert_valid(layout);
  nn_assert(ndim);
  Layout res{*this};

  // remove all dims with shape 1
  for (nn_size i = 0; i <= res.ndim - 1 && res.ndim >= 2; i++) {
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
  for (int i = static_cast<int>(res.ndim) - 2; i >= 0; i--) {
    nn_assert(res.shape[i]);
    if (res.stride[i] == res.stride[i + 1] * res.shape[i + 1]) {
      res.shape[i] *= res.shape[i + 1];
      res.stride[i] = res.stride[i + 1];
      res.remove_axis_inplace(i + 1);
    }
  }
  return res;
}

bool Layout::try_reshape(Layout &result, const Shape &tshp,
                         bool is_image) const {
  nn_assert(tshp.ndim);

  bool is_empty_shape = false;
  for (nn_size i = 0; i < tshp.ndim; ++i) {
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
    nn_assert(this->ndim >= 2 && tshp.ndim == this->ndim &&
              this->shape[ndim - 1] == tshp.shape[tshp.ndim - 2] &&
              this->shape[ndim - 2] == tshp.shape[tshp.ndim - 1]);
    for (nn_size i = 0; i < this->ndim - 2; i++) {
      if (tshp.shape[i] != this->shape[i]) return false;
    }
    result.dtype = this->dtype;
    result.Shape::operator=(tshp);
    std::swap(result.stride[this->ndim - 1], result.stride[this->ndim - 2]);
    return true;
  }

  auto cont = collapse_contiguous();
  result.dtype = this->dtype;
  result.Shape::operator=(tshp);

  if (is_empty_shape) {
    result.init_contiguous_stride();
    return true;
  }

  nn_size sdim = 0, prod = 1, cont_sdim = 0;
  for (nn_size i = 0; i < tshp.ndim; i++) {
    nn_assert(cont_sdim < cont.ndim);
    prod *= result.shape[i];
    if (prod > cont.shape[cont_sdim]) return false;

    if (prod == cont.shape[cont_sdim] &&
        (i + 1 >= tshp.ndim || tshp.shape[i + 1] != 1)) {
      auto s = cont.stride[cont_sdim];
      for (int j = i; j >= sdim; j--) {
        result.stride[j] = s;
        s *= result.shape[j];
      }
      cont_sdim++;
      sdim = i + 1;
      prod = 1;
    }
  }
  nn_assert(cont_sdim == cont.ndim);

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
