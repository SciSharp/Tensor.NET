#include <iostream>

#include "core/op/naive/ops.h"

namespace nncore {
namespace opr {
namespace naive {

#define SPECIFY_CONCAT_OP_INTERNAL(_type, _class_name)       \
  template Status _class_name::concat_internal<_type>(       \
      const std::vector<const Tensor*>& inp, _type* ptr_oup, \
      const Layout& loup, const param::concat& param);

NN_FOREACH_CTYPE_WITH_PARAM(SPECIFY_CONCAT_OP_INTERNAL, OpNaiveImpl)

template <typename T>
Status OpNaiveImpl::concat_internal(const std::vector<const Tensor*>& inp,
                                    T* oup, const Layout& loup,
                                    const param::concat& param) {
  nn_size inp_count = static_cast<nn_size>(inp.size());
  auto find_src = [&](nn_size& target_index, size_t& src_index) {
    for (nn_size i = 0; i < inp_count; i++) {
      target_index -= inp[i]->layout.shape[param.axis];
      if (target_index < 0) {
        target_index += inp[i]->layout.shape[param.axis];
        src_index = i;
        return Status::OK();
      }
    }
    return Status(StatusCategory::NUMNET, StatusCode::RUNTIME_EXCEPTION,
                  "Unexpected behaviour in cancot op.");
  };

  nn_size n = loup.total_elems();
  nn_size target_shape[4];
  for (nn_size i = 0; i < n; i++) {
    loup.offset_to_indices(i, target_shape);
    size_t src_index;
    nn_return_status_if_error(find_src(target_shape[param.axis], src_index));
    oup[i] =
        inp[src_index]
            ->ptr<T>()[inp[src_index]->layout.indices_to_offset(target_shape)];
  }
  return Status::OK();
}

}  // namespace naive
}  // namespace opr

}  // namespace nncore
