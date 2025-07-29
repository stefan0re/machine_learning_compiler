#ifndef EINSUM_BACKEND_TENSOR_OPERATION_UNARY_H
#define EINSUM_BACKEND_TENSOR_OPERATION_UNARY_H

#include <omp.h>

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "../../mini_jit/generator/Unary.h"
#include "../../tensor/tensor.h"

namespace einsum {
    namespace backend {
        class TensorOperationUnary;
    }
}  // namespace einsum

class einsum::backend::TensorOperationUnary {
   public:
    /// execution type
    enum class exec_t : uint32_t {
        seq = 0,
        prim = 1,
        shared = 2,
    };

    /// primitive type
    enum class prim_t : uint32_t {
        zero = 0,
        copy = 1,
        relu = 2,
        trans = 3,
        none = 99
    };

    /// data type
    enum class dtype_t : uint32_t {
        fp32 = 0,
        fp64 = 1
    };

    /// error codes
    enum class error_t : int32_t {
        success = 0,
        optimize_failed = 1,
        setup_failed = 2,
        compile_failed = 3,
        execute_failed = 4
    };
    /* Setup values */
    dtype_t _dtype;
    prim_t _prim_main;  // main primitive

    std::vector<exec_t> _exec_types;
    std::vector<int64_t> _dim_sizes;
    std::vector<int64_t> _strides_in0;
    std::vector<int64_t> _strides_out;

    /* Compile Values */
    std::vector<int64_t> _loop_ids;  // ids of the loops dimensions
    int64_t _id_prim_m;
    int64_t _id_prim_n;
    int64_t _id_parallel_loop = -1;

    /* Runtime Values */
    int64_t _ldi;
    int64_t _ldo;

    using kernel_t = mini_jit::generator::Unary::kernel_t;

    error_t setup(dtype_t dtype,
                  prim_t prim_main,
                  std::span<const exec_t> exec_types,
                  std::span<const int64_t> dim_sizes,
                  std::span<const int64_t> strides_in,
                  std::span<const int64_t> strides_out);

    /**
     * @brief Optimizes tensor contraction for efficient computation.
     *
     * This function performs several optimization techniques to improve
     * the performance of tensor operations, including:
     * - Dimension splitting
     * - Dimension fusion
     * - Dimension reordering
     * - Primitive identification
     * - Shared memory parallelization
     *
     * @return An error_t value indicating the success or failure of the
     *         optimization process.
     */
    error_t optimize();

    /**
     * @brief Splits dimensions to improve parallelism and memory access patterns.
     *
     */
    error_t split_dimensions();

    /**
     * @brief Fuses dimensions
     */
    error_t fuse_dimensions();

    /**
     * @brief Identifies primitives.
     *
     */
    error_t identify_primitives();

    /**
     *  @brief compile function that set all parameter for the loop over GEMM
     *
     */
    error_t compile();

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_bias Bias tensor (use nullptr if no bias).
     * @param tensor_out Output tensor.
     **/
    void execute(void const* tensor_in0,
                 void* tensor_out);

    /**
     * General-purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param id_loop      Dimension id of the loop which is executed.
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter(int64_t id_loop,
                      char const* ptr_in0,
                      char* ptr_out);

    /**
     * @brief Executes the the first loop if it is M or N dimension in parallel
     *
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter_parallel(int64_t id_loop,
                               char const* ptr_in0,
                               char const* ptr_in1,
                               char* ptr_out,
                               bool first_access,
                               bool last_access);

   private:
    // BRGEMM
    mini_jit::generator::Unary _unary;
    kernel_t _unary_kernel{nullptr};
};

#endif