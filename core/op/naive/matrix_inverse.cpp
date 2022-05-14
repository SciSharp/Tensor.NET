#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

IMPL_NAIVE_SINGLE_INPUT_INTERNAL(matrix_inverse) {
  if (linp.dtype.enumv() == DTypeEnum::Bool) {
    return Status(StatusCategory::NUMNET, StatusCode::FAIL,
                  "Cannot calculate inverse matrix for bool tensor.");
  }
  nn_size batch = 1;
  for (nn_size i = 0; i < linp.ndim - 2; i++) {
    batch *= linp.shape[i];
  }

  nn_size n = linp.shape[linp.ndim - 1];  // matrix scale

  // alloc the extra space to help the calculation
  T* rows[n];
  for (nn_size i = 0; i < n; i++) {
    // each row has 2n elements as I|X
    rows[i] = (T*)malloc(sizeof(T) * 2 * n);
  }

  nn_size offset = 0;
  for (nn_size b = 0; b < batch; b++, offset += n * n) {
    for (nn_size i = 0; i < n; i++) {
      memcpy(rows[i], ptr_inp + offset + i * n, sizeof(T) * n);
      memset(rows[i] + n, 0, sizeof(T) * n);
      rows[i][n + i] = static_cast<T>(1);
    }
    // loop on columns
    for (nn_size j = 0; j < n; j++) {
      nn_size main_elem;
      T main_elem_val = static_cast<T>(0);
      // find the main elem
      for (int i = 0; i < n; i++) {
        auto value = static_cast<T>(std::abs(rows[i][j]));
        if (value > main_elem_val) {
          main_elem = i;
          main_elem_val = value;
        }
      }

      if (main_elem_val < static_cast<T>(1e-7)) {
        return Status(StatusCategory::NUMNET, StatusCode::FAIL,
                      "Failed to calculate the inverse because the pivot value "
                      "too small (smaller than 1e-7).");
      }
      std::swap(rows[j], rows[main_elem]);

      // substract the current row from other rows
      auto cur_row_ptr = rows[j];
      main_elem_val = static_cast<T>(cur_row_ptr[j]);
      for (nn_size i = 0; i < n; i++) {
        if (i != j) {
          auto weight = -rows[i][j] / main_elem_val;
          for (nn_size k = j; k < 2 * n; k++) {
            rows[i][k] += cur_row_ptr[k] * weight;
          }
        }
      }

      // normalize the current row
      auto weight = static_cast<T>(1) / main_elem_val;
      for (nn_size k = j; k < 2 * n; k++) {
        cur_row_ptr[k] *= weight;
      }
    }

    for (nn_size i = 0; i < n; i++) {
      memcpy(ptr_oup + offset + i * n, rows[i] + n, sizeof(T) * n);
    }
  }

  // free the alloced memory
  for (nn_size i = 0; i < n; i++) {
    free(rows[i]);
    rows[i] = nullptr;
  }

  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
