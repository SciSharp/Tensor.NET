#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

using pad_param = nncore::param::pad;

template <typename T>
void exec_constant(const T* ptr_inp, T* ptr_oup, const Layout& linp,
                   const Layout& loup, const param::pad& param) {
  nn_size dst_idx[NN_MAX_NDIM], src_idx[NN_MAX_NDIM];
  nn_size n = loup.total_elems();
  memset(dst_idx, 0, sizeof dst_idx);
  dst_idx[0] = -1;

  for (nn_size i = 0; i < n; i++) {
    dst_idx[0]++;
    int flag = 0;
    nn_size constant_idx = 0;
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (dst_idx[j] == loup.shape[j]) {
        dst_idx[j] = 0;
        dst_idx[j + 1]++;
      }

      if (dst_idx[j] < param.width[j * 2]) {
        flag = 1;
        constant_idx = j;
      } else if (dst_idx[j] >= linp.shape[j] + param.width[j * 2]) {
        flag = 2;
        constant_idx = j;
      } else {
        src_idx[j] = dst_idx[j] - param.width[j * 2];
      }
    }

    if (!flag) {
      ptr_oup[loup.indices_to_offset(dst_idx)] =
          ptr_inp[linp.indices_to_offset(src_idx)];
    } else if (flag == 1) {
      ptr_oup[loup.indices_to_offset(dst_idx)] =
          static_cast<T>(param.constants[constant_idx * 2]);
    } else {
      ptr_oup[loup.indices_to_offset(dst_idx)] =
          static_cast<T>(param.constants[constant_idx * 2 + 1]);
    }
  }
}

template <typename T>
void exec_edge(const T* ptr_inp, T* ptr_oup, const Layout& linp,
               const Layout& loup, const param::pad& param) {
  nn_size dst_idx[NN_MAX_NDIM], src_idx[NN_MAX_NDIM];
  nn_size n = loup.total_elems();
  memset(dst_idx, 0, sizeof dst_idx);
  dst_idx[0] = -1;

  for (nn_size i = 0; i < n; i++) {
    dst_idx[0]++;
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (dst_idx[j] == loup.shape[j]) {
        dst_idx[j] = 0;
        dst_idx[j + 1]++;
      }

      if (dst_idx[j] < param.width[j * 2]) {
        src_idx[j] = 0;
      } else if (dst_idx[j] >= linp.shape[j] + param.width[j * 2]) {
        src_idx[j] = linp.shape[j] - 1;
      } else {
        src_idx[j] = dst_idx[j] - param.width[j * 2];
      }
    }

    ptr_oup[loup.indices_to_offset(dst_idx)] =
        ptr_inp[linp.indices_to_offset(src_idx)];
  }
}

// This implementation is not the same with that of numpy.
// The pad of the corner is not the max elements of its row
// and column in the dst tensor. Instead, it's the max value of
// the closest row and column of the src tensor.
template <typename T>
void exec_max(const T* ptr_inp, T* ptr_oup, const Layout& linp,
              const Layout& loup, const param::pad& param) {
  nn_size n_dst = loup.total_elems();
  nn_size n_src = linp.total_elems();
  nn_size dst_idx[NN_MAX_NDIM], src_idx[NN_MAX_NDIM];
  bool modify[NN_MAX_NDIM];
  T* refer[linp.ndim];

  auto get_refer_offset = [&](nn_size ignore_axis) {
    nn_size res = 0;
    for (nn_size i = 0; i < linp.ndim; i++) {
      if (i != ignore_axis) res = res * linp.shape[i] + src_idx[i];
    }
    return res;
  };

  // calculate the refers
  for (nn_size i = 0; i < linp.ndim; i++) {
    auto cur_elems = n_src / linp.shape[i];
    refer[i] = (T*)malloc(sizeof(T) * cur_elems);
    for (nn_size j = 0; j < cur_elems; j++) {
      refer[i][j] = DTypeTrait<T>::min();
    }
    for (nn_size m = 0; m < linp.shape[i]; m++) {
      memset(src_idx, 0, sizeof src_idx);
      src_idx[i] = m;
      src_idx[!i] = -1;
      for (nn_size j = 0; j < cur_elems; j++) {
        src_idx[!i]++;
        for (nn_size k = !i; k < linp.ndim; k++) {
          if (src_idx[k] == linp.shape[k]) {
            src_idx[k] = 0;
            src_idx[k + 1 + (k + 1 == i)]++;
          } else {
            break;
          }
        }
        auto temp = get_refer_offset(i);
        refer[i][temp] =
            std::max(refer[i][temp], ptr_inp[linp.indices_to_offset(src_idx)]);
      }
    }
  }
  memset(dst_idx, 0, sizeof dst_idx);
  dst_idx[0] = -1;

  // fill the values
  for (nn_size i = 0; i < n_dst; i++) {
    dst_idx[0]++;
    memset(modify, false, sizeof(bool) * NN_MAX_NDIM);
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (dst_idx[j] == loup.shape[j]) {
        dst_idx[j] = 0;
        dst_idx[j + 1]++;
      }
      if (dst_idx[j] < param.width[j * 2]) {
        modify[j] = true;
        src_idx[j] = 0;
      } else if (dst_idx[j] >= linp.shape[j] + param.width[j * 2]) {
        modify[j] = true;
        src_idx[j] = linp.shape[j] - 1;
      } else {
        src_idx[j] = dst_idx[j] - param.width[j * 2];
      }
    }
    T value = DTypeTrait<T>::min();
    bool flag = true;
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (modify[j]) {
        flag = false;
        value = std::max(value, refer[j][get_refer_offset(j)]);
      }
    }

    if (flag) {
      ptr_oup[loup.indices_to_offset(dst_idx)] =
          ptr_inp[linp.indices_to_offset(src_idx)];
    } else {
      ptr_oup[loup.indices_to_offset(dst_idx)] = value;
    }
  }

  for (nn_size i = 0; i < linp.ndim; i++) {
    free(refer[i]);
    refer[i] = nullptr;
  }
}

template <typename T>
void exec_min(const T* ptr_inp, T* ptr_oup, const Layout& linp,
              const Layout& loup, const param::pad& param) {
  nn_size n_dst = loup.total_elems();
  nn_size n_src = linp.total_elems();
  nn_size dst_idx[NN_MAX_NDIM], src_idx[NN_MAX_NDIM];
  bool modify[NN_MAX_NDIM];
  T* refer[linp.ndim];

  auto get_refer_offset = [&](nn_size ignore_axis) {
    nn_size res = 0;
    for (nn_size i = 0; i < linp.ndim; i++) {
      if (i != ignore_axis) res = res * linp.shape[i] + src_idx[i];
    }
    return res;
  };

  // calculate the refers
  for (nn_size i = 0; i < linp.ndim; i++) {
    auto cur_elems = n_src / linp.shape[i];
    refer[i] = (T*)malloc(sizeof(T) * cur_elems);
    for (nn_size j = 0; j < cur_elems; j++) {
      refer[i][j] = DTypeTrait<T>::max();
    }
    for (nn_size m = 0; m < linp.shape[i]; m++) {
      memset(src_idx, 0, sizeof src_idx);
      src_idx[i] = m;
      src_idx[!i] = -1;
      for (nn_size j = 0; j < cur_elems; j++) {
        src_idx[!i]++;
        for (nn_size k = !i; k < linp.ndim; k++) {
          if (src_idx[k] == linp.shape[k]) {
            src_idx[k] = 0;
            src_idx[k + 1 + (k + 1 == i)]++;
          } else {
            break;
          }
        }
        auto temp = get_refer_offset(i);
        refer[i][temp] =
            std::min(refer[i][temp], ptr_inp[linp.indices_to_offset(src_idx)]);
      }
    }
  }
  memset(dst_idx, 0, sizeof dst_idx);
  dst_idx[0] = -1;

  // fill the values
  for (nn_size i = 0; i < n_dst; i++) {
    dst_idx[0]++;
    memset(modify, false, sizeof(bool) * NN_MAX_NDIM);
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (dst_idx[j] == loup.shape[j]) {
        dst_idx[j] = 0;
        dst_idx[j + 1]++;
      }
      if (dst_idx[j] < param.width[j * 2]) {
        modify[j] = true;
        src_idx[j] = 0;
      } else if (dst_idx[j] >= linp.shape[j] + param.width[j * 2]) {
        modify[j] = true;
        src_idx[j] = linp.shape[j] - 1;
      } else {
        src_idx[j] = dst_idx[j] - param.width[j * 2];
      }
    }
    T value = DTypeTrait<T>::max();
    bool flag = true;
    for (nn_size j = 0; j < loup.ndim; j++) {
      if (modify[j]) {
        flag = false;
        value = std::min(value, refer[j][get_refer_offset(j)]);
      }
    }

    if (flag) {
      ptr_oup[loup.indices_to_offset(dst_idx)] =
          ptr_inp[linp.indices_to_offset(src_idx)];
    } else {
      ptr_oup[loup.indices_to_offset(dst_idx)] = value;
    }
  }

  for (nn_size i = 0; i < linp.ndim; i++) {
    free(refer[i]);
    refer[i] = nullptr;
  }
}

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(pad) {
  switch (param.mode) {
    case pad_param::Mode::Constant:
      exec_constant<T>(ptr_inp, ptr_oup, linp, loup, param);
      break;
    case pad_param::Mode::Edge:
      exec_edge<T>(ptr_inp, ptr_oup, linp, loup, param);
      break;
    case pad_param::Mode::Maximum:
      exec_max<T>(ptr_inp, ptr_oup, linp, loup, param);
      break;
    case pad_param::Mode::Minimum:
      exec_min<T>(ptr_inp, ptr_oup, linp, loup, param);
      break;

    default:
      return Status(StatusCategory::NUMNET, StatusCode::NOT_IMPLEMENTED,
                    "This mode is not implemeted so far and will be "
                    "implemented in the future version.");
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
