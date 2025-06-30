#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/backend/TensorOperation.h"
#include "../../src/mini_jit/include/gemm_ref.h"
#include "../../src/tensor/tensor.h"
#include "../test_utils/test_utils.h"

TEST_CASE("Einsum::Backend::MatMul", "[Einsum][Backend][Einsum][MatMul]") {
    std::cout << "########## Einsum binary test case 1 ##########" << std::endl;

    float* input1 = new float[10 * 20];
    test::matmul::generate_matrix(10, 20, input1, false, true);
    float* input2 = new float[20 * 30];
    test::matmul::generate_matrix(30, 20, input1, false, true);
    float* output_ref = new float[10 * 30];
    gemm_ref(input1, input2, output_ref, 10, 30, 20, 10, 20, 10);

    Tensor in_tensor1(10, 20);
    in_tensor1.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::m);
    in_tensor1.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor1.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::k);
    in_tensor1.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    Tensor in_tensor2(30, 20);
    in_tensor2.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::k);
    in_tensor2.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor2.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::n);
    in_tensor2.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);

    Tensor out_tensor(10, 30);
    out_tensor.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::m);
    out_tensor.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    out_tensor.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::n);
    out_tensor.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);

    einsum::backend::TensorOperation op;
    op.setup(einsum::backend::TensorOperation::dtype_t::fp32,
             einsum::backend::TensorOperation::prim_t::none,
             einsum::backend::TensorOperation::prim_t::gemm,
             einsum::backend::TensorOperation::prim_t::none,
             &in_tensor1, &in_tensor2, nullptr, &out_tensor);

    op.optimize();
    op.compile();
    float* output = new float[10 * 30];
    op.execute(input1, input2, nullptr, output);

    bool is_correct = test::matmul::compare_matrix(10, 30, output, output);

    REQUIRE(is_correct);
}