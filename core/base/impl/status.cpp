#include "core/base/include/status.h"

#include "core/base/include/macro.h"

namespace nncore {
Status::Status(StatusCategory category, int code, const std::string& msg) {
  nn_throw_if(code != static_cast<int>(StatusCode::OK),
              "Status OK should not be constructed by the constructor, use "
              "Status::OK() insttead");

  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code, const char* msg) {
  nn_throw_if(code != static_cast<int>(StatusCode::OK),
              "Status OK should not be constructed by the constructor, use "
              "Status::OK() insttead");

  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, "") {}

StatusCategory Status::category() const noexcept {
  return is_ok() ? StatusCategory::NONE : state_->category;
}

int Status::code() const noexcept {
  return is_ok() ? static_cast<int>(StatusCode::OK) : state_->code;
}

const std::string& Status::error_message() const noexcept {
  return is_ok() ? empty_string() : state_->msg;
}

std::string Status::to_string() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (StatusCategory::SYSTEM == state_->category) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (StatusCategory::NUMNET == state_->category) {
    result += "[Num.NET RuntimeError]";
    result += " : ";
    result += std::to_string(code());
    result += " : ";
    result += StatusCodeToString(static_cast<StatusCode>(code()));
    result += " : ";
    result += state_->msg;
  }

  return result;
}

const std::string& Status::empty_string() noexcept {
  static std::string s_empty;
  return s_empty;
}

}  // namespace nncore
