#ifndef MINI_JIT_GENERATOR_UNARY_H
#define MINI_JIT_GENERATOR_UNARY_H

#include <cstdint>

#include "../backend/Kernel.h"
#include "Util.h"

namespace mini_jit::generator {
    class Unary;
}

class mini_jit::generator::Unary {
   private:
    static mini_jit::backend::Kernel m_kernel;

    static int32_t gen_unary_zero(mini_jit::generator::Util::KernelSize kernelsize);
    static int32_t gen_unary_relu(mini_jit::generator::Util::KernelSize kernelsize);

   public:
    int32_t fops = 0;

    /// data type
    enum class dtype_t : uint32_t {
        fp32 = 0,
        fp64 = 1
    };

    /// primitive type
    enum class ptype_t : uint32_t {
        zero = 0,
        identity = 1,
        relu = 2
    };

    /// error codes
    enum class error_t : int32_t {
        success = 0
    };

    /**
     * @brief Get kernel sizes for sub-matrices.
     * @param m            Number of rows in A and B.
     * @param n            Number of columns in A and B.
     * @param kernel_sizes size for each kernel for the sub-matrices.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    static error_t get_kernel_sizes(uint32_t m,
                                    uint32_t n,
                                    Util::KernelSizes& kernel_sizes);

    /**
     * @brief Generate a kernel for a unary primitive.
     * @param m       Number of rows in A and B.
     * @param n       Number of columns in A and B.
     * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
     * @param dtype   Data type of the matrices.
     * @param ptype   Primitive type.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t generate(uint32_t m,
                     uint32_t n,
                     uint32_t trans_b,
                     dtype_t dtype,
                     ptype_t ptype);

    /*
     * Kernel type.
     * The kernel is a function that takes the following parameters:
     * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
     * - b:    Pointer to matrix B.
     * - ld_a: Leading dimension of A.
     * - ld_b: Leading dimension of B.
     */
    using kernel_t = void (*)(void const* a,
                              void* b,
                              int64_t ld_a,
                              int64_t ld_b);

    /**
     * @brief Get the generated kernel: B := op(A).
     * @return pointer to the generated kernel.
     **/
    kernel_t get_kernel() const;
};

#endif