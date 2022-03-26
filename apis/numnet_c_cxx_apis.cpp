#include "apis/numnet_c_cxx_apis.h"

#include <iostream>

using namespace nncore;

/***************Common Implementations****************/

template <typename ctype>
void print_data(const Tensor& src) {
  src.layout.dtype.assert_is_ctype<ctype>();
  nn_assert(!src.layout.is_empty(), "Cannot print an empty ndarray.");
  auto ptr = src.ptr<ctype>();
  for (int i = 0; i < src.layout.total_elems(); i++) {
    int mod = 1;
    for (int j = 0; j < src.layout.ndim; j++) {
      mod *= src.layout.shape[j];
      if (i % mod == 0) {
        std::cout << "[";
      } else {
        break;
      }
    }
    std::cout << " ";

    std::cout << ptr[i];
    if ((i + 1) % src.layout.shape[0] != 0) std::cout << ",";

    std::cout << " ";
    mod = 1;
    int hit_times = 0;
    for (int j = 0; j < src.layout.ndim; j++) {
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
      for (int j = 0; j < hit_times; j++) {
        std::cout << " ";
      }
    }
  }
  std::cout << std::endl;
}

int GetErrorCode(Status* status) {
  std::cout << "Jump in code to get error code." << std::endl;
  return status->code();
}

const char* GetErrorMessage(Status* status) {
  return status->error_message().c_str();
}

void FreeStatusMemory(Status* status) {
  free(status);
  status = nullptr;
}

/*************Operator Implementations*************/

Status* Matmul(NativeTensor* a, NativeTensor* b, NativeTensor* oup,
               param::matmul* param) {
  Tensor t_a, t_b, t_oup;
  std::cout << a->ndim << " " << b->ndim << " " << oup->ndim << std::endl;
  a->ToTensor(t_a, false);
  b->ToTensor(t_b, false);
  oup->ToTensor(t_oup, true);
  std::cout << t_a.layout.ndim << " " << t_a.layout.to_string() << std::endl;
  std::cout << t_b.layout.ndim << " " << t_b.layout.to_string() << std::endl;
  std::cout << t_oup.layout.ndim << " " << t_oup.layout.to_string()
            << std::endl;
  opr::naive::OpNaiveImpl impl;
  auto status = impl.matmul(t_a, t_b, t_oup, param::matmul());
  print_data<int>(t_a);
  print_data<int>(t_b);
  print_data<int>(t_oup);
  std::cout << t_oup.ptr<int>() << std::endl;
  Status* s = new Status(StatusCategory::NUMNET, StatusCode::MISMATCHED_DTYPE,
                         "Error!!");
  std::cout << "address: " << s << std::endl;
  return s;
  if (status.is_ok()) {
    return nullptr;
  } else {
    return &status;
  }
}