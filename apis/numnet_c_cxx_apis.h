#include "core/op/common/ops.h"
#include "core/op/naive/ops.h"

using namespace nncore;

#if __cplusplus
extern "C" {
#endif

#include <stdio.h>

/***************************Declarations**********************************/

struct NativeTensor {
  DTypeEnum dtype;
  int ndim;
  size_t offset;
  size_t* shape;
  size_t* stride;
  void* data;

  void ToTensor(Tensor& t, bool is_mutable = true) {
    t.reset_ptr(data, offset, is_mutable, false);
    t.layout.ndim = ndim;
    memcpy(t.layout.shape, shape, sizeof(size_t) * ndim);
    t.layout.dtype = DType::from_enum(dtype);
    t.layout.init_contiguous_stride();
  }
};

/***************************Common APIs**********************************/

int GetErrorCode(Status* status);

const char* GetErrorMessage(Status* status);

void FreeStatusMemory(Status* status);

/***************************Operator APIs**********************************/

Status* Matmul(NativeTensor* a, NativeTensor* b, NativeTensor* oup,
               param::matmul* param);

#if __cplusplus
}
#endif