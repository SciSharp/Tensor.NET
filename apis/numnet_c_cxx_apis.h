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
  nn_size offset;
  nn_size* shape;
  nn_size* stride;
  void* data;

  void ToTensor(Tensor& t, bool is_mutable = true) {
    t.reset_ptr(data, offset, is_mutable, false);
    t.layout.ndim = ndim;
    memcpy(t.layout.shape, shape, sizeof(nn_size) * ndim);
    memcpy(t.layout.stride, stride, sizeof(nn_size) * ndim);
    t.layout.dtype = DType::from_enum(dtype);
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

Status* Permute(NativeTensor* inp, NativeTensor* oup, param::permute* param,
                ProviderEnum provider);

#if __cplusplus
}
#endif