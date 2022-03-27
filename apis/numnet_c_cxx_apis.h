#include "core/op/common/ops.h"
#include "core/op/naive/ops.h"

using namespace nncore;

#if __cplusplus
extern "C" {
#endif

#include <stdio.h>

/***************************Declarations**********************************/

enum class ProviderEnum : int32_t { Naive = 0, ST_x86 = 1, MT_x86 = 2 };

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

opr::OpBase* GetImpl(ProviderEnum provider);

int GetErrorCode(Status* status);

const char* GetErrorMessage(Status* status);

void FreeStatusMemory(Status* status);

/***************************Operator APIs**********************************/

Status* Matmul(NativeTensor* a, NativeTensor* b, NativeTensor* oup,
               param::matmul* param, ProviderEnum provider);

#if __cplusplus
}
#endif