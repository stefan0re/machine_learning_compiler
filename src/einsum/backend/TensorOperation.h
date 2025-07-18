#ifndef EINSUM_BACKEND_TENSOR_OPERATION_H
#define EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

#include "../../mini_jit/generator/Brgemm.h"
#include "../../mini_jit/generator/Unary.h"
#include "../../tensor/tensor.h"

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
    prim_t _prim_first_touch;  // first touch primitive
    prim_t _prim_main;         // main primitive
    prim_t _prim_last_touch;   // last touch primitive

    int _prim_m_id = -1;  // id of the M dimension in input tensor 0
    int _prim_n_id = -1;  // id of the N dimension in input tensor 1
    int _prim_k_id = -1;  // id of the K dimension in input tensor 0 and 1

    int _prim_m_size = -1;
    int _prim_n_size = -1;
    int _prim_k_size = -1;

    int64_t _lda = 0;
    int64_t _ldb = 0;
    int64_t _ldc = 0;

    int _id_first_primitive_loop = -1;
    int _size_parallel_loop = 0;

    int _last_touch_id = -1;

    bool _use_bias = false;  // flag to indicate if bias is used

    std::vector<int32_t> _loop_order;

    Tensor* _tensor_in0;   // first input tensor
    Tensor* _tensor_in1;   // second input tensor (use nullptr if unary)
    Tensor* _tensor_bias;  // bias tensor (use nullptr if no bias)
    Tensor* _tensor_out;   // output tensor

    // owned storage

    using kernel_t = mini_jit::generator::Brgemm::kernel_t;

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype             Datatype of all tensor elements.
     * @param prim_first_touch  Type of the first touch primitive.
     * @param prim_main         Type of the main primitive.
     * @param prim_last_touch   Type of the last touch primitive.
     * @param in0               First input tensor.
     * @param in1               Second input tensor (use nullptr if unary).
     * @param bias              Bias tensor (use nullptr if no bias).
     * @param out               Output tensor.
     *
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t setup(dtype_t dtype,
                  prim_t prim_first_touch,
                  prim_t prim_main,
                  prim_t prim_last_touch,
                  Tensor* in0,
                  Tensor* in1,
                  Tensor* bias,
                  Tensor* out);

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
     * @param tensor_bias Bias tensor (use nullptr if no bias).
     * @param tensor_out Output tensor.
     **/
    void execute(void const* tensor_in0,
                 void const* tensor_in1,
                 void const* tensor_bias,
                 void* tensor_out);

    /**
     * General-purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param id_loop      Dimension id of the loop which is executed.
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_bias     Pointer to the bias tensor's data (use nullptr if no bias).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter(int64_t id_loop,
                      char const* ptr_in0,
                      char const* ptr_in1,
                      char const* ptr_bias,
                      char* ptr_out,
                      bool first_access,
                      bool last_access);

    /**
     * @brief Adds bias to the output tensor.
     *
     * @param ptr_in Pointer to the input tensor's data in row major.
     * @param ptr_bias Pointer to the bias tensor's data in row major.
     * @param ptr_out Pointer to the output tensor's data in row major.
     */
    void add_bias(char const* ptr_in,
                  char const* ptr_bias,
                  char* ptr_out);

    /**
     * Generates a first touch kernel with the given parameters.
     */
    void first_touch_kernel(char* ptr_in,
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
    void execute_iter_parallel(char const* ptr_in0,
                               char const* ptr_in1,
                               char const* ptr_bias,
                               char* ptr_out,
                               bool first_access,
                               bool last_access);

    void print();

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