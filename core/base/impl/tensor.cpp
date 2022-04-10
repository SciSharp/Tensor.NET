#include "core/base/include/tensor.h"

namespace nncore {
void RefPtr::reset(const void* ptr, nn_size offset, bool is_mutable,
                   bool is_owner) {
  nn_assert(m_mutable, "this RefPtr can't change.");
  *m_ref = const_cast<void*>(ptr);
  m_offset = offset;
  m_mutable = is_mutable;
  m_owned = is_owner;
}

void Tensor::reset_ptr(void* ptr, nn_size offset, bool is_mutable,
                       bool is_owner) {
  m_ref_ptr.reset(ptr, offset, is_mutable, is_owner);
}
}  // namespace nncore