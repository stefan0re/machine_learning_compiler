#ifndef EINSUM_BACKEND_TENSOR_OPERATION_H
#define EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

#include "../../mini_jit/generator/Brgemm.h"
#include "../../mini_jit/generator/Unary.h"

namespace einsum {
    namespace backend {
        class TensorOperation;
    }
}  // namespace einsum

class einsum::backend::TensorOperation {
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
        gemm = 3,
        brgemm = 4,
        none = 99
    };

    // dimension type
    enum class dim_t : uint32_t {
        c = 0,  // Dimension in all 3 tensors
        m = 1,  // Dimension in input-tensor 1 (output rows)
        n = 2,  // Dimension in input-tensor 2 (output cols)
        k = 3,  // Contraction dimension in input-tensor 1 and 2
        undefined = 99
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

    // scalars
    dtype_t _dtype;
    prim_t _prim_first_touch;
    prim_t _prim_main;
    prim_t _prim_last_touch;
    int64_t _id_first_primitive_loop;
    int64_t _num_parallel_loops;

    int64_t _id_prim_m;
    int64_t _id_prim_n;
    int64_t _id_prim_k;
    int64_t _id_prim_br;
    int64_t _id_prim_br_size = -1;

    int64_t _lda;
    int64_t _ldb;
    int64_t _ldc;
    int64_t _in0_br_stride;
    int64_t _in1_br_stride;

    // owned storage
    std::vector<dim_t> _dim_types_storage;
    std::vector<exec_t> _exec_types_storage;
    std::vector<int64_t> _dim_sizes_storage;
    std::vector<int64_t> _strides_in0_storage;
    std::vector<int64_t> _strides_in1_storage;
    std::vector<int64_t> _strides_out_storage;
    std::vector<int64_t> _loop_sizes_storage;
    std::vector<int64_t> _loop_order_storage;

    // views (spans)
    std::span<const dim_t> _dim_types;
    std::span<const exec_t> _exec_types;
    std::span<const int64_t> _dim_sizes;
    std::span<const int64_t> _strides_in0;
    std::span<const int64_t> _strides_in1;
    std::span<const int64_t> _strides_out;
    std::span<const int64_t> _loop_sizes;
    std::span<const int64_t> _loop_order;

    using kernel_t = mini_jit::generator::Brgemm::kernel_t;

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype             Datatype of all tensor elements.
     * @param prim_first_touch  Type of the first touch primitive.
     * @param prim_main         Type of the main primitive.
     * @param prim_last_touch   Type of the last touch primitive.
     * @param dim_types         Dimension type of the loops (c, m, n, or k).
     * @param exec_types        Execution type of the loops (seq, shared, or prim).
     * @param dim_sizes         Sizes of the dimensions.
     * @param strides_in0       Strides of the first input tensor.
     * @param strides_in1       Strides of the second input tensor (ignored if unary).
     * @param strides_out       Strides of the output tensor.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t setup(dtype_t dtype,
                  prim_t prim_first_touch,
                  prim_t prim_main,
                  prim_t prim_last_touch,
                  std::span<const dim_t> dim_types,
                  std::span<const exec_t> exec_types,
                  std::span<const int64_t> dim_sizes,
                  std::span<const int64_t> strides_in0,
                  std::span<const int64_t> strides_in1,
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
     * splitting M and K dims because more relevant for brgemm
     *
     */
    error_t split_dimensions();

    /**
     * Fuses dimensions to reduce the number of loops and improve efficiency.
     *
     */
    error_t fuse_dimensions();

    /**
     * Reorders dimensions to optimize memory access patterns and computation.
     *
     */
    error_t reorder_dimensions();

    /**
     * Identifies primitives for efficient computation based on tensor properties.
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
     * @param tensor_out Output tensor.
     **/
    void execute(void const* tensor_in0,
                 void const* tensor_in1,
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
                      char const* ptr_in1,
                      char* ptr_out,
                      bool first_access,
                      bool last_access,
                      int64_t loop_count);

    /**
     * Generates a first touch kernel with the given parameters.
     */
    void first_touch_kernel(char* ptr_in,
                            char* ptr_out);

    /**
     * General-purpose loop implementation featuring first and last touch operations with OMP parallelization.
     *
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter_parallel(char const* ptr_in0,
                               char const* ptr_in1,
                               char* ptr_out,
                               bool first_access,
                               bool last_access);

   private:
    // BRGEMM
    mini_jit::generator::Brgemm _brgemm;
    kernel_t _brgemm_kernel{nullptr};

    // Unary first touch
    mini_jit::generator::Unary _unary_first_touch;
    mini_jit::generator::Unary::kernel_t _unary_first_touch_kernel{nullptr};

    // Unary last touch
    mini_jit::generator::Unary _unary_last_touch;
    mini_jit::generator::Unary::kernel_t _unary_last_touch_kernel{nullptr};
};

#endif