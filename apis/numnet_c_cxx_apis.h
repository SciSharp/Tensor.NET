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
  nn_size *shape;
  nn_size *stride;
  void *data;

  void ToTensor(Tensor &t, bool is_mutable) {
    t.reset_ptr(data, offset, is_mutable, false);
    t.layout.ndim = ndim;
    memcpy(t.layout.shape, shape, sizeof(nn_size) * ndim);
    memcpy(t.layout.stride, stride, sizeof(nn_size) * ndim);
    t.layout.dtype = DType::from_enum(dtype);
  }
};

/***************************Common APIs**********************************/

opr::OpBase *GetImpl(ProviderEnum provider);

int GetErrorCode(Status *status);

const char *GetErrorMessage(Status *status);

void FreeStatusMemory(Status *status);

/***************************Operator APIs**********************************/

Status *Matmul(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
               param::matmul *param, ProviderEnum provider);

Status *Dot(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
            param::dot *param, ProviderEnum provider);

Status *BoolIndex(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
                  param::boolindex *param, ProviderEnum provider);

Status *Permute(NativeTensor *inp, NativeTensor *oup, param::permute *param,
                ProviderEnum provider);

Status *Transpose(NativeTensor *inp, NativeTensor *oup, param::transpose *param,
                  ProviderEnum provider);

Status *Repeat(NativeTensor *inp, NativeTensor *oup, param::repeat *param,
               ProviderEnum provider);

Status *Flip(NativeTensor *inp, NativeTensor *oup, param::flip *param,
             ProviderEnum provider);

Status *MatrixInverse(NativeTensor *inp, NativeTensor *oup,
                      param::matrix_inverse *param, ProviderEnum provider);

Status *Rotate(NativeTensor *inp, NativeTensor *oup, param::rotate *param,
               ProviderEnum provider);

Status *Pad(NativeTensor *inp, NativeTensor *oup, param::pad *param,
            ProviderEnum provider);

Status *Sort(NativeTensor *inp, NativeTensor *oup, param::sort *param,
             ProviderEnum provider);

Status *Onehot(NativeTensor *inp, NativeTensor *oup, param::onehot *param,
               ProviderEnum provider);

Status *Argmxx(NativeTensor *inp, NativeTensor *oup, param::argmxx *param,
               ProviderEnum provider);

Status *TypeConvert(NativeTensor *inp, NativeTensor *oup, param::convert *param,
                    ProviderEnum provider);

Status *Concat(NativeTensor *inp, int size, NativeTensor *oup,
               param::concat *param, ProviderEnum provider);

Status *Normal(NativeTensor *nt, param::normal *param, ProviderEnum provider);

Status *Uniform(NativeTensor *nt, param::uniform *param, ProviderEnum provider);

Status *Eye(NativeTensor *nt, param::eye *param, ProviderEnum provider);

Status *Fill(NativeTensor *nt, param::fill *param, ProviderEnum provider);

Status *Arange(NativeTensor *nt, param::arange *param, ProviderEnum provider);

Status *Linspace(NativeTensor *nt, param::linspace *param,
                 ProviderEnum provider);

#if __cplusplus
}
#endif