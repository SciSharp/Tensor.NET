#include "apis/numnet_c_cxx_apis.h"

#include <iostream>

using namespace nncore;
using opr::OpBase;

/***************Common Implementations****************/

opr::OpBase *GetImpl(ProviderEnum provider) {
  if (provider == ProviderEnum::Naive) {
    return opr::naive::OpNaiveImpl::get_instance();
  } else {
    return nullptr;
  }
}

template <typename ctype>
void print_data(const Tensor &src) {
  auto get_real_pos = [&src](int idx) {
    int res = 0;
    int mod = 1;
    for (int i = src.layout.ndim - 1; i >= 1; i--) mod *= src.layout.shape[i];
    for (int i = 0; i < src.layout.ndim; i++) {
      int shape = idx / mod;
      idx -= shape * mod;
      res += shape * src.layout.stride[i];
      if (i < src.layout.ndim - 1) mod /= src.layout.shape[i + 1];
    }
    return res;
  };

  src.layout.dtype.assert_is_ctype<ctype>();
  nn_assert(!src.layout.is_empty(), "Cannot print an empty ndarray.");
  auto ptr = src.ptr<ctype>();
  for (nn_size i = 0; i < src.layout.total_elems(); i++) {
    nn_size mod = 1;
    for (nn_size j = src.layout.ndim - 1; j >= 0; j--) {
      mod *= src.layout.shape[j];
      if (i % mod == 0) {
        std::cout << "[";
      } else {
        break;
      }
    }
    std::cout << " ";

    std::cout << ptr[get_real_pos(i)];

    if ((i + 1) % src.layout.shape[src.layout.ndim - 1] != 0) std::cout << ",";

    std::cout << " ";
    mod = 1;
    nn_size hit_times = 0;
    for (nn_size j = src.layout.ndim - 1; j >= 0; j--) {
      mod *= src.layout.shape[j];
      if ((i + 1) % mod == 0) {
        std::cout << "]";
        hit_times++;
      } else {
        break;
      }
    }
    if (hit_times > 0 && hit_times < src.layout.ndim) {
      std::cout << "," << std::endl;
      for (nn_size j = 0; j < src.layout.ndim - hit_times; j++) {
        std::cout << " ";
      }
    }
  }
  std::cout << std::endl;
}

int GetErrorCode(Status *status) { return status->code(); }

const char *GetErrorMessage(Status *status) {
  return status->error_message().c_str();
}

void FreeStatusMemory(Status *status) {
  free(status);
  status = nullptr;
}

/*************Operator Implementations*************/

Status *Matmul(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
               param::matmul *param, ProviderEnum provider) {
  Tensor t_a, t_b, t_oup;
  a->ToTensor(t_a, false);
  b->ToTensor(t_b, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->matmul(t_a, t_b, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Dot(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
            param::dot *param, ProviderEnum provider) {
  Tensor t_a, t_b, t_oup;
  a->ToTensor(t_a, false);
  b->ToTensor(t_b, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->dot(t_a, t_b, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *BoolIndex(NativeTensor *a, NativeTensor *b, NativeTensor *oup,
                  param::boolindex *param, ProviderEnum provider) {
  printf("Enter!\n");
  Tensor t_a, t_b, t_oup;
  a->ToTensor(t_a, false);
  b->ToTensor(t_b, false);
  oup->ToTensor(t_oup, true);
  print_data<bool>(t_b);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->boolindex(t_a, t_b, t_oup, *param);
  print_data<int>(t_oup);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Permute(NativeTensor *inp, NativeTensor *oup, param::permute *param,
                ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->permute(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Transpose(NativeTensor *inp, NativeTensor *oup, param::transpose *param,
                  ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->transpose(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Repeat(NativeTensor *inp, NativeTensor *oup, param::repeat *param,
               ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->repeat(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Flip(NativeTensor *inp, NativeTensor *oup, param::flip *param,
             ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->flip(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *MatrixInverse(NativeTensor *inp, NativeTensor *oup,
                      param::matrix_inverse *param, ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->matrix_inverse(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Rotate(NativeTensor *inp, NativeTensor *oup, param::rotate *param,
               ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->rotate(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Pad(NativeTensor *inp, NativeTensor *oup, param::pad *param,
            ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->pad(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Sort(NativeTensor *inp, NativeTensor *oup, param::sort *param,
             ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->sort(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Onehot(NativeTensor *inp, NativeTensor *oup, param::onehot *param,
               ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->onehot(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Sum(NativeTensor *inp, NativeTensor *oup, param::sum *param,
            ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->sum(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Argmxx(NativeTensor *inp, NativeTensor *oup, param::argmxx *param,
               ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->argmxx(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *TypeConvert(NativeTensor *inp, NativeTensor *oup, param::convert *param,
                    ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->convert(t_inp, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Concat(NativeTensor *inp, int size, NativeTensor *oup,
               param::concat *param, ProviderEnum provider) {
  std::vector<Tensor> tensors(size);
  std::vector<const Tensor *> tensor_ptrs(size);
  for (int i = 0; i < size; i++) {
    (inp + i)->ToTensor(tensors[i], false);
    tensor_ptrs[i] = &tensors[i];
  }
  Tensor t_oup;
  oup->ToTensor(t_oup, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->concat(tensor_ptrs, t_oup, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Normal(NativeTensor *nt, param::normal *param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->normal(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Uniform(NativeTensor *nt, param::uniform *param,
                ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->uniform(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Eye(NativeTensor *nt, param::eye *param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->eye(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Fill(NativeTensor *nt, param::fill *param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->fill(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Arange(NativeTensor *nt, param::arange *param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->arange(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}

Status *Linspace(NativeTensor *nt, param::linspace *param,
                 ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase *impl = GetImpl(provider);
  if (impl == nullptr) {
    return new Status(StatusCategory::NUMNET, StatusCode::INVALID_ARGUMENT,
                      "Unsupported provider.");
  }
  auto status = impl->linspace(t, *param);
  if (status.is_ok()) {
    return nullptr;
  } else {
    return new Status(status);
  }
}