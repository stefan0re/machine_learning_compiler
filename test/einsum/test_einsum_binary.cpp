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
    // std::cout << "########## Einsum binary test case 1 ##########" << std::endl;
    float* input1 = new float[2 * 4];
    float* input2 = new float[3 * 4];
    test::matmul::generate_matrix(2, 4, input1, false, true);
    test::matmul::generate_matrix(3, 4, input2, false, true);

    float* output_ref = new float[2 * 3];
    gemm_ref(input1, input2, output_ref, 2, 3, 4, 2, 4, 2);

    Tensor in_tensor1(2, 3, 4);
    in_tensor1.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::m);
    in_tensor1.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor1.id[0].stride = 1;
    in_tensor1.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::n);
    in_tensor1.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor1.id[1].stride = 0;
    in_tensor1.id[2].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::k);
    in_tensor1.id[2].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor1.id[2].stride = 2;
    Tensor in_tensor2(2, 3, 4);
    in_tensor2.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::m);
    in_tensor2.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor2.id[0].stride = 0;
    in_tensor2.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::n);
    in_tensor2.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor2.id[1].stride = 4;
    in_tensor2.id[2].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::k);
    in_tensor2.id[2].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    in_tensor2.id[2].stride = 1;

    Tensor out_tensor(2, 3, 4);
    out_tensor.id[0].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::m);
    out_tensor.id[0].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    out_tensor.id[0].stride = 1;
    out_tensor.id[1].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::n);
    out_tensor.id[1].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    out_tensor.id[1].stride = 2;
    out_tensor.id[2].dim_t = static_cast<int>(einsum::backend::TensorOperation::dim_t::k);
    out_tensor.id[2].exec_t = static_cast<int>(einsum::backend::TensorOperation::exec_t::seq);
    out_tensor.id[2].stride = 0;

    einsum::backend::TensorOperation op;
    op.setup(einsum::backend::TensorOperation::dtype_t::fp32,
             einsum::backend::TensorOperation::prim_t::none,
             einsum::backend::TensorOperation::prim_t::gemm,
             einsum::backend::TensorOperation::prim_t::none,
             &in_tensor1, &in_tensor2, nullptr, &out_tensor);

    op.optimize();
    op.compile();
    float* output = new float[2 * 3];
    for (size_t i = 0; i < 2 * 3; i++) {
        output[i] = 0.0f;
    }
    op.execute(input1, input2, nullptr, output);

    bool is_correct = test::matmul::compare_matrix(2, 3, output, output_ref);

    REQUIRE(is_correct);
}