#pragma once

#include "core/base/include/ndArray.h"

namespace nncore {
namespace test {
class NDArrayFactory {
 public:
  /*!
   * \brief Create an NDArray from a list
   * \param V The type of initial values
   * \param T The dtype of target NDArray
   * \return An NDArray specified by initial values, the given shape and the
   * given dtype.
   */
  template <typename V, typename T>
  static NDArray from_list(std::initializer_list<V> values, Shape shape,
                           T dtype) {
    NDArray res(shape, dtype);
    nn_assert(values.size() == shape.count(),
              "The values used to init the NDArray is not compitable with the "
              "given shape. The element count of initial values is %ld, the "
              "shape is %s, which means %ld elements.",
              values.size(), shape.to_string().c_str(), shape.count());
    // res.layout = Layout(shape, dtype);
    // res.raw_ptr = static_cast<nn_byte*>(malloc(res.layout.content_bytes()));

    auto dptr = res.ptr<typename DTypeTrait<T>::ctype>();
    for (const auto& v : values) {
      *dptr++ = typename DTypeTrait<T>::ctype(v);
    }
    return res;
  }

  /*!
   * \brief Create an empty NDArray. Note that the memory allocated for the
   * returned NDArray would not be initialized.
   * \param T The dtype of target
   * NDArray \return The empty NDArray with given shape and dtype.
   */
  template <typename T>
  static NDArray empty(Shape shape, T dtype) {
    NDArray res(shape, dtype);
    // res.layout = Layout(shape, dtype);
    // res.raw_ptr = static_cast<nn_byte*>(malloc(res.layout.content_bytes()));
    return res;
  }
};
}  // namespace test

}  // namespace nncore
