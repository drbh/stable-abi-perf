#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"

void noop(torch::Tensor const &x, int64_t n);
void zeros(torch::Tensor const &x, int64_t n);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("noop(Tensor x, int n) -> ()");
  ops.impl("noop", torch::kCUDA, &noop);
  ops.def("zeros(Tensor x, int n) -> ()");
  ops.impl("zeros", torch::kCUDA, &zeros);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
