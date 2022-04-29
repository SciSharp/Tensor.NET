#include "apis/numnet_c_cxx_apis.h"

#include <iostream>

using namespace nncore;
using opr::OpBase;

/***************Common Implementations****************/

opr::OpBase* GetImpl(ProviderEnum provider) {
  if (provider == ProviderEnum::Naive) {
    return opr::naive::OpNaiveImpl::get_instance();
  } else {
    return nullptr;
  }
}

template <typename ctype>
void print_data(const Tensor& src) {
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

int GetErrorCode(Status* status) { return status->code(); }

const char* GetErrorMessage(Status* status) {
  return status->error_message().c_str();
}

void FreeStatusMemory(Status* status) {
  free(status);
  status = nullptr;
}

/*************Operator Implementations*************/

Status* Matmul(NativeTensor* a, NativeTensor* b, NativeTensor* oup,
               param::matmul* param, ProviderEnum provider) {
  Tensor t_a, t_b, t_oup;
  a->ToTensor(t_a, false);
  b->ToTensor(t_b, false);
  oup->ToTensor(t_oup, true);
  OpBase* impl = GetImpl(provider);
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

Status* Permute(NativeTensor* inp, NativeTensor* oup, param::permute* param,
                ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase* impl = GetImpl(provider);
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

Status* Transpose(NativeTensor* inp, NativeTensor* oup, param::transpose* param,
                  ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase* impl = GetImpl(provider);
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

Status* TypeConvert(NativeTensor* inp, NativeTensor* oup, param::convert* param,
                    ProviderEnum provider) {
  Tensor t_inp, t_oup;
  inp->ToTensor(t_inp, false);
  oup->ToTensor(t_oup, true);
  OpBase* impl = GetImpl(provider);
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

Status* Normal(NativeTensor* nt, param::normal* param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase* impl = GetImpl(provider);
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

Status* Uniform(NativeTensor* nt, param::uniform* param,
                ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase* impl = GetImpl(provider);
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

Status* Eye(NativeTensor* nt, param::eye* param, ProviderEnum provider) {
  Tensor t;
  nt->ToTensor(t, true);
  OpBase* impl = GetImpl(provider);
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