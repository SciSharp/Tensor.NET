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
  nn_size *dims;

  permute(const std::vector<nn_size> &value) {
    dims = (nn_size *)malloc(sizeof(nn_size) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_size));
  }

  ~permute() {
    free(dims);
    dims = nullptr;
  }

 private:
  permute(const permute &) {}
  void operator=(const permute &) {}
};

struct repeat {
  int repeats;
  int axis;

  repeat(int repeats, int axis) : repeats(repeats), axis(axis) {}
};

struct flip {
  nn_byte *dims;

  flip(const std::vector<nn_byte> &value) {
    dims = (nn_byte *)malloc(sizeof(nn_byte) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_byte));
  }

  ~flip() {
    free(dims);
    dims = nullptr;
  }

 private:
  flip(const flip &) {}
  void operator=(const flip &) {}
};

struct matrix_inverse {};

struct rotate {
  int k;
  int dimA;
  int dimB;

  rotate(int k, int dimA, int dimB) : k(k), dimA(dimA), dimB(dimB) {
    if (dimA > dimB) std::swap(dimA, dimB);
  }
};

struct pad {
  enum Mode : int32_t {
    Constant = 1,
    Edge = 2,
    Maximum = 3,
    Minimum = 4,
    Medium = 5,
    Mean = 6
  };

  Mode mode;
  nn_size size;
  nn_size *width;
  double *constants;

  pad(Mode mode, nn_size size, const std::vector<nn_size> &width,
      const std::vector<double> &constants)
      : mode(mode), size(size) {
    this->width = (nn_size *)malloc(sizeof(nn_size) * width.size());
    memcpy(this->width, width.data(), width.size() * sizeof(nn_size));
    this->constants = (double *)malloc(sizeof(double) * constants.size());
    memcpy(this->constants, constants.data(),
           constants.size() * sizeof(double));
  };

  ~pad() {
    if (width != nullptr) {
      free(width);
      width = nullptr;
    }
    if (constants != nullptr) {
      free(constants);
      constants = nullptr;
    }
  }

 private:
  pad(const pad &) {}
  void operator=(const pad &) {}
};

struct sort {
  enum Order : int32_t { Increase = 0, Decrease = 1 };

  int axis;
  Order order;

  sort(int axis, Order order) : axis(axis), order(order) {}
};

struct arange {
  double start;
  double stop;
  double step;

  arange(int start, int stop, int step)
      : start(start), stop(stop), step(step) {}
};

struct onehot {
  int max_val;

  onehot(int max_val) : max_val(max_val) {}
};

struct sum {
  nn_byte *dims;

  sum(const std::vector<nn_byte> &value) {
    dims = (nn_byte *)malloc(sizeof(nn_byte) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_byte));
  }

  ~sum() {
    free(dims);
    dims = nullptr;
  }

 private:
  sum(const sum &) {}
  void operator=(const sum &) {}
};

struct mean {
  nn_byte *dims;

  mean(const std::vector<nn_byte> &value) {
    dims = (nn_byte *)malloc(sizeof(nn_byte) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_byte));
  }

  ~mean() {
    free(dims);
    dims = nullptr;
  }

 private:
  mean(const mean &) {}
  void operator=(const mean &) {}
};

struct max {
  nn_byte *dims;

  max(const std::vector<nn_byte> &value) {
    dims = (nn_byte *)malloc(sizeof(nn_byte) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_byte));
  }

  ~max() {
    free(dims);
    dims = nullptr;
  }

 private:
  max(const max &) {}
  void operator=(const max &) {}
};

struct min {
  nn_byte *dims;

  min(const std::vector<nn_byte> &value) {
    dims = (nn_byte *)malloc(sizeof(nn_byte) * NN_MAX_NDIM);
    memcpy(dims, value.data(), value.size() * sizeof(nn_byte));
  }

  ~min() {
    free(dims);
    dims = nullptr;
  }

 private:
  min(const min &) {}
  void operator=(const min &) {}
};

struct negative {};

struct interelem {
  enum Operation {
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Mod = 5,
    And = 6,
    Or = 7,
    Xor = 8
  };

  Operation op;

  interelem(Operation op) : op(op) {}
};

struct argmxx {
  int axis;
  bool is_max;

  argmxx(int axis, bool is_max) : axis(axis), is_max(is_max) {}
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

struct eye {
  int k;
  eye(int k) : k(k) {}
};

struct fill {
  double value;
  fill(double value) : value(value) {}
};

struct linspace {
  double start;
  double stop;
  int num;
  bool is_endpoint;

  linspace(double start, double stop, int num, bool is_endpoint)
      : start(start), stop(stop), num(num), is_endpoint(is_endpoint) {}
};

struct concat {
  int axis;

  concat(int axis) : axis(axis) {}
};

struct matmul {};

struct dot {};

struct boolindex {};
}  // namespace param

}  // namespace nncore
