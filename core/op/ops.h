#pragma once

#include <string>
#include <unordered_map>

#include "core/base/include/ndArray.h"
#include "core/macro.h"
#include "core/op/base.h"
#include "core/op/param.h"

namespace nncore {
namespace opr {
static HandleType default_handle = HandleType::Naive;

using namespace param;

// class ReshapeBase {
//  private:
//   static std::unordered_map<HandleType, ReshapeBase*> m_handle_dic;

//  public:
//   using Param = Reshape;
//   static const char* name() { return "Reshape"; }
//   static ReshapeBase* get_op(HandleType handle) {
//     nn_assert(m_handle_dic.count(handle), "Unsupported handle for op: %s.",
//               name());
//     return m_handle_dic[handle];
//   }
//   static void add_op(HandleType handle, ReshapeBase* op) {
//     if (!m_handle_dic.count(handle)) {
//       m_handle_dic.insert({handle, op});
//     }
//   }
//   virtual void exec(const NDArray& inp, const NDArray& oup,
//                     const ReshapeBase::Param& param) = 0;
//   virtual ~ReshapeBase() = default;
// };

#define DEF_OP_CLASS_1(_name)                                                 \
  class _name##Base {                                                         \
   private:                                                                   \
    inline static std::unordered_map<HandleType, _name##Base*> m_handle_dic;  \
                                                                              \
   public:                                                                    \
    using Param = _name;                                                      \
    static const char* name() { return #_name; }                              \
    static _name##Base* get_op(HandleType handle) {                           \
      nn_assert(m_handle_dic.count(handle), "Unsupported handle for op: %s.", \
                name());                                                      \
      return m_handle_dic[handle];                                            \
    }                                                                         \
    static void add_op(HandleType handle, _name##Base* op) {                  \
      if (!m_handle_dic.count(handle)) {                                      \
        m_handle_dic.insert({handle, op});                                    \
      }                                                                       \
    }                                                                         \
    virtual void exec(const NDArray& inp, const NDArray& oup,                 \
                      const _name##Base::Param& param) = 0;                   \
    virtual ~_name##Base() = default;                                         \
  };

#define DEF_OP_METHOD_1(_name)                                  \
  inline void do##_name(const NDArray& inp, const NDArray& oup, \
                        const _name##Base::Param& param,        \
                        HandleType handle = default_handle) {   \
    _name##Base::get_op(handle)->exec(inp, oup, param);         \
  }

#define DEF_OP_CLASS_2(_name)                                                 \
  class _name##Base {                                                         \
   private:                                                                   \
    inline static std::unordered_map<HandleType, _name##Base*> m_handle_dic;  \
                                                                              \
   public:                                                                    \
    using Param = _name;                                                      \
    static const char* name() { return #_name; }                              \
    static _name##Base* get_op(HandleType handle) {                           \
      nn_assert(m_handle_dic.count(handle), "Unsupported handle for op: %s.", \
                name());                                                      \
      return m_handle_dic[handle];                                            \
    }                                                                         \
    static void add_op(HandleType handle, _name##Base* op) {                  \
      if (!m_handle_dic.count(handle)) {                                      \
        m_handle_dic.insert({handle, op});                                    \
      }                                                                       \
    }                                                                         \
    virtual void exec(const NDArray& a, const NDArray& b, const NDArray& oup, \
                      const _name##Base::Param& param) = 0;                   \
    virtual ~_name##Base() = default;                                         \
  };

#define DEF_OP_METHOD_2(_name)                                               \
  inline void do##_name(const NDArray& a, const NDArray& b,                  \
                        const NDArray& oup, const _name##Base::Param& param, \
                        HandleType handle = default_handle) {                \
    _name##Base::get_op(handle)->exec(a, b, oup, param);                     \
  }

#define DEF_OP_1(_name) \
  DEF_OP_CLASS_1(_name) \
  DEF_OP_METHOD_1(_name)

#define DEF_OP_2(_name) \
  DEF_OP_CLASS_2(_name) \
  DEF_OP_METHOD_2(_name)

#define FOREACH_OP_1(cb) cb(Reshape) cb(Transpose)
FOREACH_OP_1(DEF_OP_1)
#undef FOREACH_OP_1

#define FOREACH_OP_2(cb) cb(MatMul) cb(Dot)
FOREACH_OP_2(DEF_OP_2)
#undef FOREACH_OP_1

}  // namespace opr

}  // namespace nncore
