#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

template <typename T>
void quick_sort_increase(nn_size base_offset, nn_size stride, nn_size l,
                         nn_size r, T* ptr_oup) {
  if (l >= r) return;
  auto refer = ptr_oup[base_offset + (l + r + 1) / 2 * stride];
  auto a = l - 1, b = r + 1;
  while (a < b) {
    a++;
    while (ptr_oup[base_offset + a * stride] < refer) a++;
    b--;
    while (ptr_oup[base_offset + b * stride] > refer) b--;
    if (a < b)
      std::swap(ptr_oup[base_offset + a * stride],
                ptr_oup[base_offset + b * stride]);
  }
  quick_sort_increase<T>(base_offset, stride, l, a - 1, ptr_oup);
  quick_sort_increase<T>(base_offset, stride, a, r, ptr_oup);
}

template <typename T>
void quick_sort_decrease(nn_size base_offset, nn_size stride, nn_size l,
                         nn_size r, T* ptr_oup) {
  if (l >= r) return;
  auto refer = ptr_oup[base_offset + (l + r + 1) / 2 * stride];
  auto a = l - 1, b = r + 1;
  while (a < b) {
    a++;
    while (ptr_oup[base_offset + a * stride] > refer) a++;
    b--;
    while (ptr_oup[base_offset + b * stride] < refer) b--;
    if (a < b)
      std::swap(ptr_oup[base_offset + a * stride],
                ptr_oup[base_offset + b * stride]);
  }
  quick_sort_decrease<T>(base_offset, stride, l, a - 1, ptr_oup);
  quick_sort_decrease<T>(base_offset, stride, a, r, ptr_oup);
}

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(sort) {
  nn_size n = linp.total_elems();
  nn_size src_idx[NN_MAX_NDIM];

  // copy all the data to target tensor first
  for (nn_size i = 0; i < n; i++) {
    loup.offset_to_indices(i, src_idx);
    ptr_oup[i] = ptr_inp[linp.indices_to_offset(src_idx)];
  }

  // quick sort
  memset(src_idx, 0, sizeof src_idx);
  src_idx[!param.axis] = -1;
  auto other_elems = n / loup.shape[param.axis];
  auto axis_shape = loup.shape[param.axis];
  auto stride = loup.stride[param.axis];

  auto increase_idx = [&]() {
    src_idx[!param.axis]++;
    for (nn_size i = !param.axis; i < linp.ndim; i++) {
      if (i == param.axis) continue;
      if (src_idx[i] == linp.shape[i]) {
        src_idx[i] = 0;
        src_idx[i + 1 + (i + 1 == param.axis)]++;
      }
    }
    return linp.indices_to_offset(src_idx);
  };

  if (param.order == param::sort::Order::Increase) {
    for (auto i = 0; i < other_elems; i++) {
      auto base_offset = increase_idx();
      quick_sort_increase<T>(base_offset, stride, 0, axis_shape - 1, ptr_oup);
    }
  } else {
    for (auto i = 0; i < other_elems; i++) {
      auto base_offset = increase_idx();
      quick_sort_decrease<T>(base_offset, stride, 0, axis_shape - 1, ptr_oup);
    }
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
