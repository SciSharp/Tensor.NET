#pragma once

#include <iostream>
#include <memory>

#include "core/op/naive/matmul/opr_impl.h"

namespace nncore {

  #define REGISTER_METHOD_1(_name) \
            void _name(const NDArray& inp, const NDArray& oup, \
                        const _name::Param& param,        \
                        HandleType handle = default_handle);

class Methods{
protected:
  HandleType m_type;
public:
  HandleType type() const{
    return m_type;
  }
};

#define ADD_HANDLE(_name, _impl, _handle) \
  opr::##_name##Base::add_op(_handle, _impl::get_instance())

/*
 * This is a method which must be called at the beginning of the program.
 * It registered implemention of different handles for each opr.
 */
static void init_handles() {
  // ADD_HANDLE(MatMul, MatMulImpl, HandleType::Naive);
  opr::MatMulBase::add_op(HandleType::Naive,
                          opr::naive::MatMulImpl::get_instance());
  std::cout << "init!" << std::endl;
}

class HandleInitializer {
  static HandleInitializer instance;
  HandleInitializer() { init_handles(); }
};

// inline HandleInitializer HandleInitializer::instance;
}  // namespace nncore

namespace nncore {
namespace nn {
static nncore::HandleType get_default_handle() {
  return nncore::opr::default_handle;
}
static void set_default_handle(nncore::HandleType handle) {
  nncore::opr::default_handle = handle;
}
}  // namespace nn
}  // namespace nncore