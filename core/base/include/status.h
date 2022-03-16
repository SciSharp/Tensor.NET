#pragma once

#include <memory>
#include <string>

namespace nncore {

#define nn_return_status_if_error(_expr) \
  Status _status = _expr;                \
  if (!_status.is_ok()) return _status;

#define nn_throw_if_status_fail(_expr) \
  do {                                 \
    Status _status = (expr);           \
    if ((!_status.IsOK())) {           \
      throw _status.to_string();       \
    }                                  \
  } while (0)

enum StatusCategory { NONE = 0, SYSTEM = 1, NUMNET = 2 };

/**
   Error code for ONNXRuntime.
*/
enum StatusCode {
  OK = 0,
  FAIL = 1,
  INVALID_ARGUMENT = 2,
  MISMATCHED_SHAPE = 3,
  MISMATCHED_DTYPE = 4,
  ENGINE_ERROR = 5,
  RUNTIME_EXCEPTION = 6,
  INVALID_PROTOBUF = 7,
  NOT_IMPLEMENTED = 8
};

constexpr const char* StatusCodeToString(StatusCode status) noexcept {
  switch (status) {
    case StatusCode::OK:
      return "SUCCESS";
    case StatusCode::FAIL:
      return "FAIL";
    case StatusCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case StatusCode::MISMATCHED_SHAPE:
      return "MISMATCHED_SHAPE";
    case StatusCode::MISMATCHED_DTYPE:
      return "MISMATCHED_DTYPE";
    case StatusCode::ENGINE_ERROR:
      return "ENGINE_ERROR";
    case StatusCode::RUNTIME_EXCEPTION:
      return "RUNTIME_EXCEPTION";
    case StatusCode::INVALID_PROTOBUF:
      return "INVALID_PROTOBUF";
    case StatusCode::NOT_IMPLEMENTED:
      return "NOT_IMPLEMENTED";
    default:
      return "GENERAL ERROR";
  }
}

class [[nodiscard]] Status {
 public:
  Status() noexcept = default;

  Status(StatusCategory category, int code, const std::string& msg);

  Status(StatusCategory category, int code, const char* msg);

  Status(StatusCategory category, int code);

  Status(const Status& other)
      : state_((other.state_ == nullptr) ? nullptr : new State(*other.state_)) {
  }

  Status& operator=(const Status& other) {
    if (state_ != other.state_) {
      if (other.state_ == nullptr) {
        state_.reset();
      } else {
        state_.reset(new State(*other.state_));
      }
    }
    return *this;
  }

  Status(Status &&) = default;
  Status& operator=(Status&&) = default;
  ~Status() = default;

  bool is_ok() const { return (state_ == nullptr); }

  int code() const noexcept;

  StatusCategory category() const noexcept;

  const std::string& error_message() const noexcept;

  std::string to_string() const;

  bool operator==(const Status& other) const {
    return (this->state_ == other.state_) || (to_string() == other.to_string());
  }

  bool operator!=(const Status& other) const { return !(*this == other); }

  static Status OK() { return Status(); }

 private:
  static const std::string& empty_string() noexcept;

  struct State {
    State(StatusCategory cat0, int code0, const std::string& msg0)
        : category(cat0), code(code0), msg(msg0) {}

    State(StatusCategory cat0, int code0, const char* msg0)
        : category(cat0), code(code0), msg(msg0) {}

    const StatusCategory category;
    const int code;
    const std::string msg;
  };

  // As long as Code() is OK, state_ == nullptr.
  std::unique_ptr<State> state_;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status) {
  return out << status.to_string();
}
}  // namespace nncore