#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/backend/TensorOperation.h"

using namespace einsum::backend;

void example1_2_ref(float* in0,
                    float* in1,
                    float* out) {
    for (size_t M = 0; M < 32; M++) {
        for (size_t N = 0; N < 32; N++) {
            for (size_t K = 0; K < 8; K++) {
                for (size_t m = 0; m < 32; m++) {
                    for (size_t n = 0; n < 32; n++) {
                        for (size_t k = 0; k < 32; k++) {
                            // compute the index for in0, in1 and out
                            size_t idx_in0 = M * 8192 + K * 1024 + m * 1 + n * 0 + k * 32;
                            size_t idx_in1 = N * 8192 + K * 1024 + m * 0 + n * 32 + k * 1;
                            size_t idx_out = M * 32768 + N * 1024 + K * 0 + m * 1 + n * 32 + k * 0;

                            // perform the operation
                            out[idx_out] += in0[idx_in0] * in1[idx_in1];
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Einsum::Backend::TensorOperation 1. pbtc example", "First pbtc example") {
    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute reference
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }
    std::cout << "  Total error first example: " << error << std::endl;
    REQUIRE(error < 1e-5);

    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}

TEST_CASE("Einsum::Backend::TensorOperation 2. pbtc example", "Second pbtc example") {
    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute reference
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }
    std::cout << "  Total error second example: " << error << std::endl;
    REQUIRE(error < 1e-5);

    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}

TEST_CASE("Einsum::Backend::TensorOperation 3 pbtc example", "Third pbtc example") {
    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::relu,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute reference
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // apply ReLU to reference output
    for (size_t i = 0; i < size_out; i++) {
        tensor_out_ref[i] = std::max(0.0f, tensor_out_ref[i]);
    }
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
        if (std::abs(tensor_out[i] - tensor_out_ref[i]) > 1e-5) {
            std::cout << "Error at index " << i << ": " << tensor_out[i] << " vs " << tensor_out_ref[i] << std::endl;
            break;
        }
    }
    std::cout << "  Total error third example: " << error << std::endl;
    REQUIRE(error < 1e-5);
    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}