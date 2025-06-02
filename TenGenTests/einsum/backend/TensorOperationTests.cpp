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

TEST_CASE("Example 1", "[Einsum][Backend][TensorOperation]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor3(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor4;

    // create xtensor tensors with the same shape
    // Note: xtensor uses row major layout by default, so we specify it explicitly
    xt::xtensor<float, 4, xt::layout_type::column_major> xt_tensor1 = xt::xtensor<float, 4, xt::layout_type::column_major>::from_shape({32, 8, 32, 32});
    xt::xtensor<float, 4, xt::layout_type::column_major> xt_tensor2 = xt::xtensor<float, 4, xt::layout_type::column_major>::from_shape({32, 8, 32, 32});

    // copy the data from the TenGen tensors to the xtensor tensors
    std::copy(tenGenTensor1.exportPointer(), tenGenTensor1.exportPointer() + tenGenTensor1.size, xt_tensor1.data());
    std::copy(tenGenTensor2.exportPointer(), tenGenTensor2.exportPointer() + tenGenTensor2.size, xt_tensor2.data());

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
                            std::vector<int64_t>{8192, 0, 1024, 1, 0, 32},
                            std::vector<int64_t>{0, 8192, 1024, 0, 32, 1},
                            std::vector<int64_t>{32768, 1024, 0, 1, 32, 0});

    to.execute(tenGenTensor1.exportPointer(), tenGenTensor2.exportPointer(), tenGenTensor3.exportPointer());

    //
    //
    //

    // abdc, ebfd -> acef
    auto result = xt::linalg::tensordot(xt_tensor1, xt_tensor2, std::vector<size_t>{1, 2}, std::vector<size_t>{1, 3});
    // aefc
    auto out = xt::transpose(result, {0, 2, 3, 1});

    //
    //
    //

    // make sure the result is in column major order
    xt::xtensor<float, 4, xt::layout_type::column_major> output(out);

    // copy the result to the TenGen tensor
    std::copy(output.data(), output.data() + output.size(), tenGenTensor4.exportPointer());

    // check if the TenGen tensor is equal to the output tensor
    REQUIRE(tenGenTensor3 == tenGenTensor4);
}