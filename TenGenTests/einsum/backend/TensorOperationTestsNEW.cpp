#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <span>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>

#include "TenGen.h"
#include "TenGen/einsum/backend/TensorOperation.h"
#include "TenGen/tensor/Tensor.h"
#include "TenGen/types/Structs.h"
#include "TenGen/types/Types.h"
#include "TenGenTests/TenGenTestsHelper.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using TensorOperation = TenGen::Einsum::Backend::TensorOperation;

/*  Example 1 (einsum expression) abdc, ebfd -> aefc
    - dtype	FP32
    - prim_first_touch	None
    - prim_main	GEMM
    - prim_last_touch	None
    - dim_types  ( M, N, K, M, N, K )
    - exec_types ( Seq, Seq, Seq, Prim, Prim, Prim )
    - dim_sizes	 ( 32, 32, 8, 32, 32, 32 )
    - strides_in0	( 8192, 0, 1024, 1, 0, 32 )
    - strides_in1	( 0, 8192, 1024, 0, 32, 1 )
    - strides_out	( 32768, 1024, 0, 1, 32, 0 )

    aufbau input1: 32x8x32x32 -> abdc
    aufbau input2: 32x8x32x32 -> ebfd
    aufbau output: 32x32x32x32 -> aefc
*/

/*
- The math is preserved under reordering.
- Precision may vary slightly.
- Performance can vary a lot.
*/

TEST_CASE("Example 1", "[Einsum][Backend][TensorOperation]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor3(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor4;

    // Define the shape an size of the tensors
    std::vector<std::int64_t> shape_in = {32, 8, 32, 32};
    std::vector<std::int64_t> strides_in1 = {8192, 0, 1024, 1, 0, 32};
    std::vector<std::int64_t> strides_in2 = {0, 8192, 1024, 0, 32, 1};

    std::vector<std::int64_t> shape_out = {32, 32, 32, 32};
    std::vector<std::int64_t> strides_out = {32768, 1024, 0, 1, 32, 0};

    // wrap the TenGen tensors into xtensor views
    auto xtensor_1 = xt::adapt(tenGenTensor1.exportPointer(), 32 * 8 * 32 * 32, xt::no_ownership(), shape_in, strides_in1);
    auto xtensor_2 = xt::adapt(tenGenTensor2.exportPointer(), 32 * 8 * 32 * 32, xt::no_ownership(), shape_in, strides_in2);

    // Contract over axes: {1, 2} in both
    auto temp = xt::linalg::tensordot(xtensor_1, xtensor_2, {1, 2}, {1, 3});
    auto output = xt::transpose(temp, {0, 1, 3, 2});

    //
    //
    //

    TensorOperation to;

    // Setup the tensor operation
    auto l_error = to.setup(dtype_t::fp32,
                            prim_t::none,
                            prim_t::gemm,
                            prim_t::none,
                            std::vector<dim_t>{dim_t::m, dim_t::n, dim_t::k, dim_t::m, dim_t::n, dim_t::k},
                            std::vector<exec_t>{exec_t::seq, exec_t::seq, exec_t::seq, exec_t::prim, exec_t::prim, exec_t::prim},
                            std::vector<int64_t>{32, 32, 8, 32, 32, 32},
                            strides_in1,
                            strides_in2,
                            strides_out);

    to.execute(tenGenTensor1.exportPointer(), tenGenTensor2.exportPointer(), tenGenTensor3.exportPointer());

    //
    //
    //

    //
    //
    //

    // copy the result to the TenGen tensor
    std::copy(output.data(), output.data() + output.size(), tenGenTensor4.exportPointer());

    // check if the TenGen tensor is equal to the output tensor
    REQUIRE(tenGenTensor3 == tenGenTensor4);
}