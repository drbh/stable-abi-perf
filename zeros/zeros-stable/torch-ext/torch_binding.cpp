#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "registration.h"

void noop(torch::stable::Tensor const &x, int64_t n);
void zeros(torch::stable::Tensor const &x, int64_t n);

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("noop(Tensor x, int n) -> ()");
  ops.def("zeros(Tensor x, int n) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("noop", TORCH_BOX(&noop));
  ops.impl("zeros", TORCH_BOX(&zeros));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
