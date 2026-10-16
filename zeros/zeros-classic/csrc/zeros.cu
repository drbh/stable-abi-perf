#include <torch/all.h>

void noop(torch::Tensor const &x, int64_t n) {}

// n back-to-back torch::zeros allocations shaped like x
void zeros(torch::Tensor const &x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) torch::zeros(x.sizes(), x.options());
}
