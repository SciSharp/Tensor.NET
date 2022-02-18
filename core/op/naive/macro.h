#pragma once

namespace nncore {
namespace naive {
#define FOREACH_OPR_TYPE_CHECK(cb) cb(int) cb(float) cb(double)
}  // namespace naive

}  // namespace nncore
