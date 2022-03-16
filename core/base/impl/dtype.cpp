#include "core/base/include/dtype.h"

namespace nncore {
DTypeEnum DType::enumv() const { return m_trait->enumv; }

bool DType::is_same_with(const DType &rhs) const {
  return m_trait->enumv == rhs.m_trait->enumv;
}

void DType::assert_is(const DType &rhs) const {
  nn_assert(m_trait->enumv == rhs.m_trait->enumv);
}

bool DType::operator==(const DType &rhs) const {
  return m_trait->enumv == rhs.m_trait->enumv;
}

bool DType::operator!=(const DType &rhs) const {
  return m_trait->enumv != rhs.m_trait->enumv;
}

DType DType::from_enum(DTypeEnum enumv) {
  switch (enumv) {
#define cb(_dt)        \
  case DTypeEnum::_dt: \
    return dtype::_dt();
    NN_FOREACH_DTYPE(cb)
#undef cb
  }
  nn_throw(
      "Failed to create Dtype from enum because of invalid DTypeEnum value.");
}

namespace dtype {
#define DECLARE_SM_TRAIT(_name)    \
  DType::Trait _name::sm_trait = { \
      DTypeTrait<_name>::name, DTypeTrait<_name>::size_log, DTypeEnum::_name};

NN_FOREACH_DTYPE(DECLARE_SM_TRAIT)
#undef DECLARE_SM_TRAIT
}  // namespace dtype

}  // namespace nncore