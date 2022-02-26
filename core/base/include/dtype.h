#pragma once

#include <stdint.h>

#include <string>

#include "core/macro.h"

namespace nncore {
enum class DTypeEnum : uint32_t { Invalid, Int32, Float32, Float64, Bool };

class DType {
 protected:
  struct Trait {
    const char *const name;
    const uint16_t size_log;  //!< log2 of sizeof(dt) for non-lowbit
    DTypeEnum enumv;
  };
  Trait *m_trait;
  explicit DType(Trait *t) : m_trait(t) {}

 public:
  DType() : m_trait(nullptr) {}

  DTypeEnum enumv() const;

  const char *name() const { return m_trait->name; }

  /*!
   * \brief size of this data type in bytes
   */
  size_t size() const { return 1 << m_trait->size_log; }

  bool is_valid() { return m_trait != nullptr; }

  template <typename T>
  inline bool is_ctype() const;

  template <typename T>
  inline void assert_is_ctype() const;

  bool is_same_with(const DType &rhs) const;

  void assert_is(const DType &rhs) const;

  bool operator==(const DType &rhs) const;
  bool operator!=(const DType &rhs) const;

  static DType from_enum(DTypeEnum enumv);
};

template <class T>
class DTypeTrait;

namespace dtype {

//! Map basic data types
#define nn_byte unsigned char
#define nn_int32 int
#define nn_float32 float
#define nn_float64 double
#define nn_bool bool

//! Define min and max value of data types
#define max_val_nn_byte 255
#define min_val_nn_byte 0
#define max_val_nn_int32 2147483647
#define min_val_nn_int32 -2147483648
#define max_val_nn_float32 3.40282346638528859811704183484516925e+38F
#define min_val_nn_float32 -3.40282346638528859811704183484516925e+38F
#define max_val_nn_float64 1.79769313486231570E+308
#define min_val_nn_float64 -1.79769313486231570E+308
#define max_val_nn_bool 1
#define min_val_nn_bool 0

#define NN_FOREACH_DTYPE(cb) cb(Int32) cb(Float32) cb(Float64) cb(Bool)

#define NN_FOREACH_CTYPE_WITH_PARAM(cb, ...)            \
  cb(nn_int32, __VA_ARGS__) cb(nn_float32, __VA_ARGS__) \
      cb(nn_float64, __VA_ARGS__) cb(nn_bool, __VA_ARGS__)

#define NN_DECLARE_DTYPE(_name)      \
  class _name final : public DType { \
   private:                          \
    static Trait sm_trait;           \
                                     \
   public:                           \
    _name() : DType(&sm_trait) {}    \
  };
NN_FOREACH_DTYPE(NN_DECLARE_DTYPE)
#undef NN_DECLARE_DTYPE

#define NN_DEF_DTYPE_TRAIT_BASIC(_name, _ctype, _bits) \
  static constexpr const char *name = #_name;          \
  using ctype = _ctype;                                \
  using dtype = ::nncore::dtype::_name;                \
  static constexpr uint16_t size_log = _bits;          \
  static constexpr DTypeEnum enumv = DTypeEnum::_name;

}  // namespace dtype

#define NN_FOREACH_DTYPE_TRAIT(cb)                    \
  cb(Int32, nn_int32, 32) cb(Float32, nn_float32, 32) \
      cb(Float64, nn_float64, 64) cb(Bool, nn_bool, 8)

#define NN_DEF_DTYPE_TRAIT(_name, _ctype, _bits)    \
  template <>                                       \
  struct DTypeTrait<dtype::_name> {                 \
    NN_DEF_DTYPE_TRAIT_BASIC(_name, _ctype, _bits)  \
    static ctype min() { return max_val_##_ctype; } \
    static ctype max() { return min_val_##_ctype; } \
  };

NN_FOREACH_DTYPE_TRAIT(NN_DEF_DTYPE_TRAIT)
#undef NN_DEF_DTYPE_TRAIT

// alias DTypeTrait for ctypes
#define IMPL(_obj)                                  \
  template <>                                       \
  struct DTypeTrait<DTypeTrait<dtype::_obj>::ctype> \
      : public DTypeTrait<dtype::_obj> {};

NN_FOREACH_DTYPE(IMPL)
#undef IMPL

template <typename T>
inline bool DType::is_ctype() const {
  return (typename DTypeTrait<T>::dtype()).enumv() == m_trait->enumv;
}

template <typename T>
inline void DType::assert_is_ctype() const {
  nn_assert((typename DTypeTrait<T>::dtype()).enumv() == m_trait->enumv);
}

}  // namespace nncore