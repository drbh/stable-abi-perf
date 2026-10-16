#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

void noop(torch::stable::Tensor const &x, int64_t n) {}

// n back-to-back torch::stable::new_zeros allocations shaped like x
void zeros(torch::stable::Tensor const &x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) torch::stable::new_zeros(x, x.sizes());
}
