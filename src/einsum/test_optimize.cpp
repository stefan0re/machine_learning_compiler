#include <chrono>
#include <iostream>
#include <span>

#include "../mini_jit/generator/Brgemm.h"
#include "backend/TensorOperation.h"
#include "include/einsum_ref.h"

using namespace einsum::backend;

#define DEBUG

bool check_diff(float* i_ten_1,
                float* i_ten_2,
                size_t i_size) {
    bool l_equal = true;
    double l_max_diff = 0.0;
    for (size_t i = 0; i < i_size; ++i) {
        if (std::abs(i_ten_1[i] - i_ten_2[i]) > 1e-3f) {
            l_equal = false;
            l_max_diff = std::max(l_max_diff, static_cast<double>(std::abs(i_ten_1[i] - i_ten_2[i])));
        }
    }
    if (l_equal) {
        return true;
    } else {
        std::cout << "Max difference: " << l_max_diff << std::endl;
        return false;
    }
}

/** Settings 1
 * dim_types	( M, N, K )
 * exec_types	( Seq, Seq, Seq )
 * dim_sizes	( 1600, 1600, 1600 )
 * strides_in0	( 1, 0, 1600 )
 * strides_in1	( 0, 1600, 1 )
 * strides_out	( 1, 1600, 0 )
 */

void run_setting_1() {
    float* l_ten_1 = new float[1600 * 1600];
    float* l_ten_2 = new float[1600 * 1600];
    float* l_out_scalar = new float[1600 * 1600];
    float* l_out_einsum_1 = new float[1600 * 1600];

    srand48(0);

    for (size_t i = 0; i < 1600 * 1600; ++i) {
        l_ten_1[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    for (size_t i = 0; i < 1600 * 1600; ++i) {
        l_ten_2[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    // Initialize output tensors to zero
    for (size_t i = 0; i < 1600 * 1600; ++i) {
        l_out_scalar[i] = 0.0f;
        l_out_einsum_1[i] = 0.0f;
    }
    TensorOperation l_tensor_op;
    TensorOperation::dtype_t l_dtype = TensorOperation::dtype_t::fp32;
    TensorOperation::prim_t l_prim_first_touch = TensorOperation::prim_t::none;
    TensorOperation::prim_t l_prim_main = TensorOperation::prim_t::gemm;
    TensorOperation::prim_t l_prim_last_touch = TensorOperation::prim_t::none;
    std::vector<TensorOperation::dim_t> l_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> l_exec_types = {TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};
    std::vector<int64_t> l_dim_sizes = {1600, 1600, 1600};

    std::vector<int64_t> l_strides_in0 = {1, 0, 1600};  // M, N, K
    std::vector<int64_t> l_strides_in1 = {0, 1600, 1};  // N, K, M
    std::vector<int64_t> l_strides_out = {1, 1600, 0};  // M, N, K

    // Setup the tensor operation
    auto l_error = l_tensor_op.setup(l_dtype,
                                     l_prim_first_touch,
                                     l_prim_main,
                                     l_prim_last_touch,
                                     std::span<const TensorOperation::dim_t>(l_dim_types),
                                     std::span<const TensorOperation::exec_t>(l_exec_types),
                                     std::span<const int64_t>(l_dim_sizes),
                                     std::span<const int64_t>(l_strides_in0),
                                     std::span<const int64_t>(l_strides_in1),
                                     std::span<const int64_t>(l_strides_out));

    l_tensor_op.optimize();

#ifdef DEBUG
    // print all necessary information
    std::cout << "***********************" << std::endl;
    std::cout << "TensorOperation setup:" << std::endl;
    std::cout << "  dtype: " << static_cast<int>(l_dtype) << std::endl;
    std::cout << "  prim_first_touch: " << static_cast<int>(l_prim_first_touch) << std::endl;
    std::cout << "  prim_main: " << static_cast<int>(l_prim_main) << std::endl;
    std::cout << "  prim_last_touch: " << static_cast<int>(l_prim_last_touch) << std::endl;
    std::cout << "  id_first_primitive_loop: " << l_tensor_op._id_first_primitive_loop << std::endl;
    std::cout << "  id_prim_m: " << l_tensor_op._id_prim_m << std::endl;
    std::cout << "  id_prim_n: " << l_tensor_op._id_prim_n << std::endl;
    std::cout << "  id_prim_k: " << l_tensor_op._id_prim_k << std::endl;
    std::cout << "  id_prim_br: " << l_tensor_op._id_prim_br << std::endl;

    // print all strides
    std::cout << "  strides_in0: ";
    for (const auto& stride : l_tensor_op._strides_in0) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_in1: ";
    for (const auto& stride : l_tensor_op._strides_in1) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_out: ";
    for (const auto& stride : l_tensor_op._strides_out) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_types: ";
    for (const auto& type : l_tensor_op._dim_types) {
        std::cout << static_cast<int>(type) << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_sizes: ";
    for (const auto& size : l_tensor_op._dim_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print execution types
    std::cout << "  exec_types: ";
    for (const auto& exec_type : l_tensor_op._exec_types) {
        std::cout << static_cast<int>(exec_type) << " ";
    }
    std::cout << std::endl;

    // print loop sizes
    std::cout << "  loop_sizes: ";
    for (const auto& size : l_tensor_op._loop_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print loop order
    std::cout << "  loop_order: ";
    for (const auto& order : l_tensor_op._loop_order) {
        std::cout << order << " ";
    }
    std::cout << std::endl;

#endif

    l_tensor_op.compile();

    l_tensor_op.execute(l_ten_1, l_ten_2, l_out_einsum_1);

    // clean up
    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_1;
    std::cout << "Setting 1 completed." << std::endl;
}

/**
 * Setting 2
 * dtype	FP32
 * prim_first_touch	None
 * prim_main	GEMM
 * prim_last_touch	None
 * dim_types	( M, N, K, M, N, K )
 * exec_types	( Seq, Seq, Seq, Prim, Prim, Prim )
 * dim_sizes	( 32, 32, 8, 32, 32, 32 )
 * strides_in0	( 8192, 0, 1024, 1, 0, 32 )
 * strides_in1	( 0, 8192, 1024, 0, 32, 1 )
 * strides_out	( 32768, 1024, 0, 1, 32, 0 )
 */

void run_setting_2() {
    float* l_ten_1 = new float[32 * 8 * 32 * 32];
    float* l_ten_2 = new float[32 * 8 * 32 * 32];
    float* l_out_scalar = new float[32 * 32 * 32 * 32];
    float* l_out_einsum_1 = new float[32 * 32 * 32 * 32];

    srand48(0);
    for (size_t i = 0; i < 32 * 8 * 32 * 32; ++i) {
        l_ten_1[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    for (size_t i = 0; i < 32 * 8 * 32 * 32; ++i) {
        l_ten_2[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    // Initialize output tensors to zero
    for (size_t i = 0; i < 32 * 32 * 32 * 32; ++i) {
        l_out_scalar[i] = 0.0f;
        l_out_einsum_1[i] = 0.0f;
    }

    TensorOperation l_tensor_op;
    TensorOperation::dtype_t l_dtype = TensorOperation::dtype_t::fp32;
    TensorOperation::prim_t l_prim_first_touch = TensorOperation::prim_t::none;
    TensorOperation::prim_t l_prim_main = TensorOperation::prim_t::gemm;
    TensorOperation::prim_t l_prim_last_touch = TensorOperation::prim_t::none;
    std::vector<TensorOperation::dim_t> l_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> l_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq};
    std::vector<int64_t> l_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> l_strides_in0 = {8192, 0, 1024, 1, 0, 32};   // M, N, K
    std::vector<int64_t> l_strides_in1 = {0, 8192, 1024, 0, 32, 1};   // N, K, M
    std::vector<int64_t> l_strides_out = {32768, 1024, 0, 1, 32, 0};  // M, N, K
    // Setup the tensor operation
    auto l_error = l_tensor_op.setup(l_dtype,
                                     l_prim_first_touch,
                                     l_prim_main,
                                     l_prim_last_touch,
                                     std::span<const TensorOperation::dim_t>(l_dim_types),
                                     std::span<const TensorOperation::exec_t>(l_exec_types),
                                     std::span<const int64_t>(l_dim_sizes),
                                     std::span<const int64_t>(l_strides_in0),
                                     std::span<const int64_t>(l_strides_in1),
                                     std::span<const int64_t>(l_strides_out));
    l_tensor_op.optimize();

#ifdef DEBUG
    // print all necessary information
    std::cout << "***********************" << std::endl;
    std::cout << "TensorOperation setup:" << std::endl;
    std::cout << "  dtype: " << static_cast<int>(l_dtype) << std::endl;
    std::cout << "  prim_first_touch: " << static_cast<int>(l_prim_first_touch) << std::endl;
    std::cout << "  prim_main: " << static_cast<int>(l_prim_main) << std::endl;
    std::cout << "  prim_last_touch: " << static_cast<int>(l_prim_last_touch) << std::endl;
    std::cout << "  id_prim_m: " << l_tensor_op._id_prim_m << std::endl;
    std::cout << "  id_prim_n: " << l_tensor_op._id_prim_n << std::endl;
    std::cout << "  id_prim_k: " << l_tensor_op._id_prim_k << std::endl;
    std::cout << "  id_prim_br: " << l_tensor_op._id_prim_br << std::endl;

    // print all strides
    std::cout << "  strides_in0: ";
    for (const auto& stride : l_tensor_op._strides_in0) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_in1: ";
    for (const auto& stride : l_tensor_op._strides_in1) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_out: ";
    for (const auto& stride : l_tensor_op._strides_out) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_types: ";
    for (const auto& type : l_tensor_op._dim_types) {
        std::cout << static_cast<int>(type) << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_sizes: ";
    for (const auto& size : l_tensor_op._dim_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print execution types
    std::cout << "  exec_types: ";
    for (const auto& exec_type : l_tensor_op._exec_types) {
        std::cout << static_cast<int>(exec_type) << " ";
    }
    std::cout << std::endl;

    // print loop sizes
    std::cout << "  loop_sizes: ";
    for (const auto& size : l_tensor_op._loop_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print loop order
    std::cout << "  loop_order: ";
    for (const auto& order : l_tensor_op._loop_order) {
        std::cout << order << " ";
    }
    std::cout << std::endl;

#endif

    l_tensor_op.compile();

    l_tensor_op.execute(l_ten_1, l_ten_2, l_out_einsum_1);

    // cleab up
    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_1;
    std::cout << "Setting 2 completed." << std::endl;
}

/** Setting 3
 * dim_types	( M, M, N, N, K, K )
 * exec_types	( Seq, Seq, Seq, Seq, Seq, Seq )
 * dim_sizes	( 64, 25, 64, 25, 64, 25 )
 * strides_in0	( 25, 1, 0, 0, 40000, 1600 )
 * strides_in1	( 0, 0, 40000, 1600, 25, 1 )
 * strides_out	( 25, 1, 40000, 1600, 0, 0 )
 */

void run_setting_3() {
    float* l_ten_1 = new float[64 * 25 * 64 * 25];
    float* l_ten_2 = new float[64 * 25 * 64 * 25];
    float* l_out_scalar = new float[64 * 25 * 64 * 25];
    float* l_out_einsum_1 = new float[64 * 25 * 64 * 25];

    srand48(0);
    for (size_t i = 0; i < 64 * 25 * 64 * 25; ++i) {
        l_ten_1[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    for (size_t i = 0; i < 64 * 25 * 64 * 25; ++i) {
        l_ten_2[i] = (10 * static_cast<float>(drand48())) - 5.0;
    }
    // Initialize output tensors to zero
    for (size_t i = 0; i < 64 * 25 * 64 * 25; ++i) {
        l_out_scalar[i] = 0.0f;
        l_out_einsum_1[i] = 0.0f;
    }

    TensorOperation l_tensor_op;
    TensorOperation::dtype_t l_dtype = TensorOperation::dtype_t::fp32;
    TensorOperation::prim_t l_prim_first_touch = TensorOperation::prim_t::none;
    TensorOperation::prim_t l_prim_main = TensorOperation::prim_t::gemm;
    TensorOperation::prim_t l_prim_last_touch = TensorOperation::prim_t::none;
    std::vector<TensorOperation::dim_t> l_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> l_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq};
    std::vector<int64_t> l_dim_sizes = {64, 25, 64, 25, 64, 25};
    std::vector<int64_t> l_strides_in0 = {25, 1, 0, 0, 40000, 1600};  // M, N, K
    std::vector<int64_t> l_strides_in1 = {0, 0, 40000, 1600, 25, 1};  // N, K, M
    std::vector<int64_t> l_strides_out = {25, 1, 40000, 1600, 0, 0};  // M, N, K

    // Setup the tensor operation
    auto l_error = l_tensor_op.setup(l_dtype,
                                     l_prim_first_touch,
                                     l_prim_main,
                                     l_prim_last_touch,
                                     std::span<const TensorOperation::dim_t>(l_dim_types),
                                     std::span<const TensorOperation::exec_t>(l_exec_types),
                                     std::span<const int64_t>(l_dim_sizes),
                                     std::span<const int64_t>(l_strides_in0),
                                     std::span<const int64_t>(l_strides_in1),
                                     std::span<const int64_t>(l_strides_out));

    l_tensor_op.optimize();

#ifdef DEBUG
    // print all necessary information
    std::cout << "***********************" << std::endl;
    std::cout << "TensorOperation setup:" << std::endl;
    std::cout << "  dtype: " << static_cast<int>(l_dtype) << std::endl;
    std::cout << "  prim_first_touch: " << static_cast<int>(l_prim_first_touch) << std::endl;
    std::cout << "  prim_main: " << static_cast<int>(l_prim_main) << std::endl;
    std::cout << "  prim_last_touch: " << static_cast<int>(l_prim_last_touch) << std::endl;
    std::cout << "  id_prim_m: " << l_tensor_op._id_prim_m << std::endl;
    std::cout << "  id_prim_n: " << l_tensor_op._id_prim_n << std::endl;
    std::cout << "  id_prim_k: " << l_tensor_op._id_prim_k << std::endl;
    std::cout << "  id_prim_br: " << l_tensor_op._id_prim_br << std::endl;

    // print all strides
    std::cout << "  strides_in0: ";
    for (const auto& stride : l_tensor_op._strides_in0) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_in1: ";
    for (const auto& stride : l_tensor_op._strides_in1) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  strides_out: ";
    for (const auto& stride : l_tensor_op._strides_out) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_types: ";
    for (const auto& type : l_tensor_op._dim_types) {
        std::cout << static_cast<int>(type) << " ";
    }
    std::cout << std::endl;
    std::cout << "  dim_sizes: ";
    for (const auto& size : l_tensor_op._dim_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print execution types
    std::cout << "  exec_types: ";
    for (const auto& exec_type : l_tensor_op._exec_types) {
        std::cout << static_cast<int>(exec_type) << " ";
    }
    std::cout << std::endl;

    // print loop sizes
    std::cout << "  loop_sizes: ";
    for (const auto& size : l_tensor_op._loop_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // print loop order
    std::cout << "  loop_order: ";
    for (const auto& order : l_tensor_op._loop_order) {
        std::cout << order << " ";
    }
    std::cout << std::endl;

#endif

    l_tensor_op.compile();
    l_tensor_op.execute(l_ten_1, l_ten_2, l_out_einsum_1);

    // clean up
    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_1;
    std::cout << "Setting 3 completed." << std::endl;
}

int main() {
    std::cout << "Running first setting." << std::endl;

    // run_setting_1();
    // run_setting_2();
    run_setting_3();

    std::cout << "Test passed successfully!" << std::endl;
    return 0;
}