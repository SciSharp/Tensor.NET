#pragma once

namespace nncore {

enum class HandleType { Naive, CPU };

class OpBase {
 public:
  explicit OpBase(HandleType handle);
  virtual ~OpBase() = default;

  HandleType handle() { return m_handle; }

 protected:
  HandleType m_handle;
};

}  // namespace nncore
