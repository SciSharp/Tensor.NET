#include "core/base/include/dtype.h"
#include "core/base/include/macro.h"
#include "core/base/include/status.h"

namespace nncore {
namespace opr {

inline DTypeEnum deduce_double_input_op(DTypeEnum a, DTypeEnum b) {
  if (a == DTypeEnum::Float64 || b == DTypeEnum::Float64) {
    return DTypeEnum::Float64;
  } else if (a == DTypeEnum::Float32 || b == DTypeEnum::Float32) {
    return DTypeEnum::Float32;
  } else if (a == DTypeEnum::Int64 || b == DTypeEnum::Int64) {
    return DTypeEnum::Int64;
  } else if (a == DTypeEnum::Int32 || b == DTypeEnum::Int32) {
    return DTypeEnum::Int32;
  } else if (a == DTypeEnum::Bool || b == DTypeEnum::Bool) {
    return DTypeEnum::Bool;
  } else {
    nn_throw("Type deduce failed.");
  }
}

#define DEDUCE_CASE(_value, _A, _B, _C, _name, ...)              \
  case (_value):                                                 \
    nn_return_status_if_error((_name##_internal<_A, _B, _C>(     \
        a.ptr<_A>(), b.ptr<_B>(), oup.ptr<_C>(), __VA_ARGS__))); \
    break;

#define DOUBLE_INPUT_DTYPE_DEDUCE(_DA, _DB, _name, ...)                       \
  {                                                                           \
    int _value = static_cast<int>(_DA) * 10 + static_cast<int>(_DB);          \
    switch (_value) {                                                         \
      DEDUCE_CASE(11, nn_int32, nn_int32, nn_int32, _name, __VA_ARGS__)       \
      DEDUCE_CASE(12, nn_int32, nn_float32, nn_float32, _name, __VA_ARGS__)   \
      DEDUCE_CASE(13, nn_int32, nn_float64, nn_float64, _name, __VA_ARGS__)   \
      DEDUCE_CASE(21, nn_float32, nn_int32, nn_float32, _name, __VA_ARGS__)   \
      DEDUCE_CASE(22, nn_float32, nn_float32, nn_float32, _name, __VA_ARGS__) \
      DEDUCE_CASE(23, nn_float32, nn_float64, nn_float64, _name, __VA_ARGS__) \
      DEDUCE_CASE(31, nn_float64, nn_int32, nn_float64, _name, __VA_ARGS__)   \
      DEDUCE_CASE(32, nn_float64, nn_float32, nn_float64, _name, __VA_ARGS__) \
      DEDUCE_CASE(33, nn_float64, nn_float64, nn_float64, _name, __VA_ARGS__) \
                                                                              \
      DEDUCE_CASE(14, nn_int32, nn_int64, nn_int64, _name, __VA_ARGS__)       \
      DEDUCE_CASE(41, nn_int64, nn_int32, nn_int64, _name, __VA_ARGS__)       \
      DEDUCE_CASE(44, nn_int64, nn_int64, nn_int64, _name, __VA_ARGS__)       \
                                                                              \
      DEDUCE_CASE(42, nn_int64, nn_float32, nn_float32, _name, __VA_ARGS__)   \
      DEDUCE_CASE(43, nn_int64, nn_float64, nn_float64, _name, __VA_ARGS__)   \
      DEDUCE_CASE(24, nn_float32, nn_int64, nn_float32, _name, __VA_ARGS__)   \
      DEDUCE_CASE(34, nn_float64, nn_int64, nn_float64, _name, __VA_ARGS__)   \
                                                                              \
      DEDUCE_CASE(15, nn_int32, nn_bool, nn_int32, _name, __VA_ARGS__)        \
      DEDUCE_CASE(51, nn_bool, nn_int32, nn_int32, _name, __VA_ARGS__)        \
      DEDUCE_CASE(25, nn_float32, nn_bool, nn_float32, _name, __VA_ARGS__)    \
      DEDUCE_CASE(52, nn_bool, nn_float32, nn_float32, _name, __VA_ARGS__)    \
      DEDUCE_CASE(35, nn_float64, nn_bool, nn_float64, _name, __VA_ARGS__)    \
      DEDUCE_CASE(53, nn_bool, nn_float64, nn_float64, _name, __VA_ARGS__)    \
      DEDUCE_CASE(45, nn_int64, nn_bool, nn_int64, _name, __VA_ARGS__)        \
      DEDUCE_CASE(54, nn_bool, nn_int64, nn_int64, _name, __VA_ARGS__)        \
      DEDUCE_CASE(55, nn_bool, nn_bool, nn_bool, _name, __VA_ARGS__)          \
      default:                                                                \
        return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_DTYPE,   \
                      "Type deduce failed.");                                 \
    }                                                                         \
  }

#define FOREACH_DOUBLE_INPUT_TYPE_PAIR(cb, ...)        \
  cb(nn_int32, nn_int32, nn_int32, __VA_ARGS__);       \
  cb(nn_int32, nn_float32, nn_float32, __VA_ARGS__);   \
  cb(nn_int32, nn_float64, nn_float64, __VA_ARGS__);   \
  cb(nn_float32, nn_int32, nn_float32, __VA_ARGS__);   \
  cb(nn_float32, nn_float32, nn_float32, __VA_ARGS__); \
  cb(nn_float32, nn_float64, nn_float64, __VA_ARGS__); \
  cb(nn_float64, nn_int32, nn_float64, __VA_ARGS__);   \
  cb(nn_float64, nn_float32, nn_float64, __VA_ARGS__); \
  cb(nn_float64, nn_float64, nn_float64, __VA_ARGS__); \
  cb(nn_int32, nn_int64, nn_int64, __VA_ARGS__);       \
  cb(nn_int64, nn_int32, nn_int64, __VA_ARGS__);       \
  cb(nn_int64, nn_int64, nn_int64, __VA_ARGS__);       \
  cb(nn_int64, nn_float32, nn_float32, __VA_ARGS__);   \
  cb(nn_int64, nn_float64, nn_float64, __VA_ARGS__);   \
  cb(nn_float32, nn_int64, nn_float32, __VA_ARGS__);   \
  cb(nn_float64, nn_int64, nn_float64, __VA_ARGS__);   \
  cb(nn_int32, nn_bool, nn_int32, __VA_ARGS__);        \
  cb(nn_bool, nn_int32, nn_int32, __VA_ARGS__);        \
  cb(nn_float32, nn_bool, nn_float32, __VA_ARGS__);    \
  cb(nn_bool, nn_float32, nn_float32, __VA_ARGS__);    \
  cb(nn_float64, nn_bool, nn_float64, __VA_ARGS__);    \
  cb(nn_bool, nn_float64, nn_float64, __VA_ARGS__);    \
  cb(nn_int64, nn_bool, nn_int64, __VA_ARGS__);        \
  cb(nn_bool, nn_int64, nn_int64, __VA_ARGS__);        \
  cb(nn_bool, nn_bool, nn_bool, __VA_ARGS__);

#define TYPE_SELECT_SELF_MODIFY(_type, _name)                      \
  if (t.layout.dtype.is_ctype<_type>()) {                          \
    nn_return_status_if_error(                                     \
        _name##_internal<_type>(t.ptr<_type>(), t.layout, param)); \
    return Status::OK();                                           \
  }

#define TYPE_SELECT_SINGLE_INPUT(_type, _name, _loup)             \
  if (_loup.dtype.is_ctype<_type>()) {                            \
    nn_return_status_if_error(_name##_internal<_type>(            \
        inp.ptr<_type>(), oup.ptr<_type>(), linp, _loup, param)); \
    return Status::OK();                                          \
  }

#define TYPE_SELECT_CONCAT(_type, _loup)                              \
  if (_loup.dtype.is_ctype<_type>()) {                                \
    nn_return_status_if_error(                                        \
        concat_internal<_type>(inp, oup.ptr<_type>(), _loup, param)); \
    return Status::OK();                                              \
  }

#define SPECIFY_SELF_MODIFY_OP_INTERNAL(_type, _class_name, _op_name) \
  template Status _class_name::_op_name##_internal<_type>(            \
      _type * t, const Layout& layout, const param::_op_name& param);

#define SPECIFY_SINGLE_OUTPUT_OP_INTERNAL(_type, _class_name, _op_name)     \
  template Status _class_name::_op_name##_internal<_type>(                  \
      const _type* inp, _type* oup, const Layout& linp, const Layout& loup, \
      const param::_op_name& param);

#define SPECIFY_DOUBLE_OUTPUT_OP_INTERNAL(_typeA, _typeB, _typeC, _class_name, \
                                          _op_name)                            \
  template Status _class_name::_op_name##_internal<_typeA, _typeB, _typeC>(    \
      const _typeA* ptr_a, const _typeB* ptr_b, _typeC* ptr_oup,               \
      const Layout& la, const Layout& lb, const Layout& loup,                  \
      const param::_op_name& param)

#define SPECIFY_CONVERT_OP_INTERNAL(_typeA, _typeB, _, _class_name) \
  template Status _class_name::convert_internal<_typeA, _typeB>(    \
      const _typeA* ptr_inp, _typeB* ptr_oup, const Layout& linp,   \
      const Layout& loup, const param::convert& param)

#define CONVERT_CASE(_value, _A, _B, ...)                                      \
  case (_value):                                                               \
    nn_return_status_if_error((                                                \
        convert_internal<_A, _B>(inp.ptr<_A>(), oup.ptr<_B>(), __VA_ARGS__))); \
    break;

#define TYPE_CONVERT_DEDUCE(_DA, _DB, ...)                                  \
  {                                                                         \
    int _value = static_cast<int>(_DA) * 10 + static_cast<int>(_DB);        \
    switch (_value) {                                                       \
      CONVERT_CASE(11, nn_int32, nn_int32, __VA_ARGS__)                     \
      CONVERT_CASE(12, nn_int32, nn_float32, __VA_ARGS__)                   \
      CONVERT_CASE(13, nn_int32, nn_float64, __VA_ARGS__)                   \
      CONVERT_CASE(21, nn_float32, nn_int32, __VA_ARGS__)                   \
      CONVERT_CASE(22, nn_float32, nn_float32, __VA_ARGS__)                 \
      CONVERT_CASE(23, nn_float32, nn_float64, __VA_ARGS__)                 \
      CONVERT_CASE(31, nn_float64, nn_int32, __VA_ARGS__)                   \
      CONVERT_CASE(32, nn_float64, nn_float32, __VA_ARGS__)                 \
      CONVERT_CASE(33, nn_float64, nn_float64, __VA_ARGS__)                 \
                                                                            \
      CONVERT_CASE(14, nn_int32, nn_int64, __VA_ARGS__)                     \
      CONVERT_CASE(41, nn_int64, nn_int32, __VA_ARGS__)                     \
      CONVERT_CASE(44, nn_int64, nn_int64, __VA_ARGS__)                     \
                                                                            \
      CONVERT_CASE(42, nn_int64, nn_float32, __VA_ARGS__)                   \
      CONVERT_CASE(43, nn_int64, nn_float64, __VA_ARGS__)                   \
      CONVERT_CASE(24, nn_float32, nn_int64, __VA_ARGS__)                   \
      CONVERT_CASE(34, nn_float64, nn_int64, __VA_ARGS__)                   \
                                                                            \
      CONVERT_CASE(15, nn_int32, nn_bool, __VA_ARGS__)                      \
      CONVERT_CASE(51, nn_bool, nn_int32, __VA_ARGS__)                      \
      CONVERT_CASE(25, nn_float32, nn_bool, __VA_ARGS__)                    \
      CONVERT_CASE(52, nn_bool, nn_float32, __VA_ARGS__)                    \
      CONVERT_CASE(35, nn_float64, nn_bool, __VA_ARGS__)                    \
      CONVERT_CASE(53, nn_bool, nn_float64, __VA_ARGS__)                    \
      CONVERT_CASE(45, nn_int64, nn_bool, __VA_ARGS__)                      \
      CONVERT_CASE(54, nn_bool, nn_int64, __VA_ARGS__)                      \
      CONVERT_CASE(55, nn_bool, nn_bool, __VA_ARGS__)                       \
      default:                                                              \
        return Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_DTYPE, \
                      "Type convert failed.");                              \
    }                                                                       \
  }

}  // namespace opr
}  // namespace nncore