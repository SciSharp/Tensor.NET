#pragma once

#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "core/base/include/dtype.h"
#include "core/base/include/macro.h"

namespace nncore {

#define nn_size int

struct Shape {
  static constexpr nn_size MAX_NDIM = NN_MAX_NDIM;

  nn_size shape[MAX_NDIM];
  nn_size ndim = 0;

  // ctor
  Shape() = default;
  Shape(const Shape &rhs) = default;
  Shape(nn_size *init_shape, int ndim);
  Shape(const std::vector<nn_size> &init_shape);
  Shape(const std::initializer_list<nn_size> &init_shape);

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
  nn_size total_elems() const;

  nn_size &operator[](nn_size i) { return shape[i]; }
  nn_size operator[](nn_size i) const { return shape[i]; }

  std::string to_string() const;
};

struct Layout : public Shape {
  DType dtype;
  nn_size stride[MAX_NDIM];

  // ctor
  Layout();
  Layout(const Layout &rhs) = default;
  Layout(const DType &dtype);
  Layout(const Shape &shape, const DType &dtype);
  Layout(const Shape &shape, const std::vector<nn_size> &stride,
         const DType &dtype);

  bool is_same_layout(const Layout &rhs) const;

  /*
   * \brief Return if the two layouts are equivalent, which means that one could
   * directly reshape to the other without changing the arrangement of elements.
   *
   * For instance, {1, 1, 2, 3} and {1, 2, 3} are equivalent. Instead, {1, 2, 3}
   * and {3, 2, 1} are not, though {1, 2, 3} could be reshaped to {3, 2, 1}.
   */
  bool is_equivalent_layout(const Layout &rhs) const;

  /*
   * \brief Convert offset to indices of the current layout.
   * \param indices An array which has same or larger size than the ndim of the
   * current layout.
   */
  void offset_to_indices(nn_size offset, nn_size *indices) const;

  /*
   * \brief Convert offset to indices of the current layout.
   * \param dst An array which has same or larger size than the ndim of the
   * current layout.
   */
  nn_size indices_to_offset(nn_size *indices) const;

  std::string to_string() const;

  nn_size content_bytes() const;

  /* =================== inplace modifiers =================== */

  /*!
   * \brief init stride to be contiguous
   *
   * Use current shape and format
   *
   * \return total number of elements
   */
  nn_size init_contiguous_stride();

  /*!
   * \brief init stride to be contiguous by first assigning shape
   *
   * Use current format.
   */
  nn_size init_contiguous_stride(const Shape &shape);

  /*!
   * \brief inplace version of remove_axis
   */
  void remove_axis_inplace(nn_size idx);

  /*!
   * \brief add an axis before given *axis* with given shape and stride
   *
   * Other shapes and strides would not be changed.
   */
  void add_axis_inplace(nn_size axis, nn_size shape, nn_size stride);

  /*!
   * \brief add an axis before given *axis*, with shape 1 and contiguous
   *      stride
   */
  void add_axis_cont_inplace(nn_size axis) {
    add_axis_inplace(axis, 1, stride[axis] * shape[axis]);
  }

  /*!
   * \brief modify data type of the layout inplace
   *
   * By the way this API will modify the format according to the data type
   */
  void modify_dtype_inplace(DType dtype);

  /* =================== generate new layout =================== */

  /**
   * \brief Returns the layout with permuted dimensions.
   *
   * example:
   *  (2, 0, 1) -> AxBxC to CxAxB
   */
  Layout dimshuffle(const std::vector<nn_size> &dims) const;

  /**
   * \brief Remove an axis from the layout by moving later shape/stride
   *      elements earlier. No extra check is performed.
   */
  Layout remove_axis(nn_size idx) const;

  /**
   * \brief Returns a different view.
   *
   * \throw TensorReshapeError if no stride exists for target shape.
   */
  Layout reshape(const Shape &shape, bool is_image = false) const;

  /*!
   * \brief try to reshape to another view; return whether these two shapes
   *      are compatible
   * \return true iff there exists target stride so this layout can be
   *      converted to target shape and the elements can match.
   */
  bool try_reshape(Layout &output, const Shape &shape,
                   bool is_image = false) const;

  /*!
   * \brief Broadcast on dims with shape == 1 to match target *shape*.
   * \throw TensorReshapeError if could not be satisfied
   */
  Layout broadcast(const Shape &target) const;
  void broadcast_inplace(const Shape &target);

  /*!
   * \brief Collapse consecutive axes with contiguous layout together
   *
   * This transforms the tensor into a canonized form. For empty tensors or
   * scalar, the result would always be a one-dimensional empty or scalar,
   * with stride being 1.
   */
  Layout collapse_contiguous() const;

  bool is_contiguous() const;
};
}  // namespace nncore
