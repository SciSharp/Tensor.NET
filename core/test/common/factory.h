#pragma once

#include "core/base/include/tensor.h"

namespace nncore {
namespace test {
class NDArrayFactory {
 public:
  /*!
   * \brief Create an Tensor from a list
   * \param V The type of initial values
   * \param T The dtype of target Tensor
   * \return An Tensor specified by initial values, the given shape and the
   * given dtype.
   */
  template <typename V, typename T>
  static Tensor from_list(std::initializer_list<V> values, Shape shape,
                          T dtype) {
    Tensor res(shape, dtype);
    nn_assert(static_cast<nn_size>(values.size()) == shape.total_elems(),
              "The values used to init the Tensor is not compitable with the "
              "given shape. The element count of initial values is %ld, the "
              "shape is %s, which means %d elements.",
              values.size(), shape.to_string().c_str(), shape.total_elems());
    // res.layout = Layout(shape, dtype);
    // res.raw_ptr = static_cast<nn_byte*>(malloc(res.layout.content_bytes()));

    auto dptr = res.ptr<typename DTypeTrait<T>::ctype>();
    for (const auto& v : values) {
      *dptr++ = typename DTypeTrait<T>::ctype(v);
    }
    return res;
  }

  /*!
   * \brief Create an empty Tensor. Note that the memory allocated for the
   * returned Tensor would not be initialized.
   * \param T The dtype of target
   * Tensor \return The empty Tensor with given shape and dtype.
   */
  template <typename T>
  static Tensor empty(Shape shape, T dtype) {
    Tensor res(shape, dtype);
    // res.layout = Layout(shape, dtype);
    // res.raw_ptr = static_cast<nn_byte*>(malloc(res.layout.content_bytes()));
    return res;
  }
};
}  // namespace test

}  // namespace nncore
