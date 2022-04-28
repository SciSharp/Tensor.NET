#pragma once

#include <memory.h>

#include "core/base/include/layout.h"

namespace nncore {
namespace param {

struct transpose {
  nn_size dimA;
  nn_size dimB;

  transpose(nn_size a, nn_size b) : dimA(a), dimB(b) {}
};

struct permute {
  nn_size* dims;

  permute(const std::vector<nn_size>& value) {
    dims = (nn_size*)malloc(sizeof(nn_size) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_size));
  }

  ~permute() {
    free(dims);
    dims = nullptr;
  }

 private:
  permute(const permute&) {}
  void operator=(const permute&) {}
};

struct convert {
  DTypeEnum target_type;

  convert(DTypeEnum target_type) : target_type(target_type) {}
};

struct normal {
  double avg;
  double std;

  normal(double avg, double std) : avg(avg), std(std) {}
};

struct uniform {
  double min_value;
  double max_value;
  uniform(double min_value, double max_value)
      : min_value(min_value), max_value(max_value) {}
};

struct matmul {};

struct dot {};
}  // namespace param

}  // namespace nncore
