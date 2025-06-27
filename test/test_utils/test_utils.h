#ifndef EINSUM_TREES_EINSUM_TREE_H
#define EINSUM_TREES_EINSUM_TREE_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../../tensor/tensor.h"
#include "../backend/TensorOperation.h"

namespace test {
    namespace matmul {
        void generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero = false, bool visualization = false);
        bool compare_matrix(uint32_t height, uint32_t width, float* M, float* C);
    }  // namespace matmul
}  // namespace test

#endif