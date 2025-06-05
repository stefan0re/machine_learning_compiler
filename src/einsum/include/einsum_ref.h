#ifndef EINSUM_INCLUDE_EINSUM_REF_H
#define EINSUM_INCLUDE_EINSUM_REF_H

#include <cstdint>

#include "../../mini_jit/include/gemm_ref.h"
#include "../backend/TensorOperation.h"

std::vector<int64_t> prime_factors(int64_t n);

int64_t find_new_size(std::vector<int64_t> const& i_sizes);

#endif  // EINSUM_INCLUDE_EINSUM_REF_H