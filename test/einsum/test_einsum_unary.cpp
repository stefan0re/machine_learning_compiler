#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/backend/TensorOperationUnary.h"

using namespace einsum::backend;

/**
 * abc->abc
 * dim_types = C
 * exec_types = ( Seq, Prim, Prim)
 * dim_sizes = (2, 3, 2)
 * strides_in0 = (6, 2, 1)
 * strides_out =  (6, 2, 1)
 * prim_type = relu
 */
TEST_CASE("Einsum::Backend::TensorOperationUnary Relu", "TEST 1") {
    TensorOperationUnary tensor_op;

    std::vector<TensorOperationUnary::exec_t> i_exec_types = {TensorOperationUnary::exec_t::seq,
                                                              TensorOperationUnary::exec_t::prim,
                                                              TensorOperationUnary::exec_t::prim};
    std::vector<int64_t> i_dim_sizes = {2, 3, 2};
    std::vector<int64_t> i_strides_in0 = {6, 2, 1};
    std::vector<int64_t> i_strides_out = {6, 2, 1};

    std::span<TensorOperationUnary::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_out_span(i_strides_out);
    tensor_op.setup(TensorOperationUnary::dtype_t::fp32,
                    TensorOperationUnary::prim_t::relu,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_out_span);
    tensor_op.compile();

    // create in0 and out tensor
    int64_t size_in0 = 2 * 3 * 2;
    int64_t size_out = 2 * 3 * 2;
    float* tensor_in0 = new float[size_in0];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }
    // execute reference
    for (size_t i = 0; i < size_in0; i++) {
        tensor_out_ref[i] = std::max(0.0f, tensor_in0[i]);
    }
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_out);
    // check if output is correct
    for (size_t i = 0; i < size_out; i++) {
        if (tensor_out[i] != tensor_out_ref[i]) {
            std::cerr << "Error: Output does not match reference at index " << i << ": "
                      << tensor_out[i] << " != " << tensor_out_ref[i] << std::endl;
            REQUIRE(tensor_out[i] == tensor_out_ref[i]);
        }
    }
}

/**
 * abc->cab
 * dim_types = C
 * exec_types = ( Seq, Seq, Seq)
 * dim_sizes = (2, 3, 2)
 * strides_in0 = (6, 2, 1)
 * strides_out =  (3, 1, 6)
 * prim_type = trans
 */
TEST_CASE("Einsum::Backend::TensorOperationUnary Reorder", "TEST 2") {
    TensorOperationUnary tensor_op;

    std::vector<TensorOperationUnary::exec_t> i_exec_types = {TensorOperationUnary::exec_t::seq,
                                                              TensorOperationUnary::exec_t::seq,
                                                              TensorOperationUnary::exec_t::seq};
    std::vector<int64_t> i_dim_sizes = {2, 3, 2};
    std::vector<int64_t> i_strides_in0 = {6, 2, 1};
    std::vector<int64_t> i_strides_out = {3, 1, 6};

    std::span<TensorOperationUnary::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_out_span(i_strides_out);
    tensor_op.setup(TensorOperationUnary::dtype_t::fp32,
                    TensorOperationUnary::prim_t::trans,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_out_span);
    tensor_op.compile();

    // create in0 and out tensor
    int64_t size_in0 = 2 * 3 * 2;
    int64_t size_out = 2 * 3 * 2;
    float* tensor_in0 = new float[size_in0];
    float* tensor_out = new float[size_out];
    float tensor_out_ref[12] = {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};

    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = i + 1;
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
    };

    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_out);

    for (size_t i = 0; i < size_out; i++) {
        std::cout << tensor_out[i] << " ";
    }
    std::cout << std::endl;

    // check if output is correct
    for (size_t i = 0; i < size_out; i++) {
        if (tensor_out[i] != tensor_out_ref[i]) {
            std::cerr << "Error: Output does not match reference at index " << i << ": "
                      << tensor_out[i] << " != " << tensor_out_ref[i] << std::endl;
            REQUIRE(tensor_out[i] == tensor_out_ref[i]);
        }
    }
}