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
        relu = 2,
        trans = 3
    };

    /// error codes
    enum class error_t : int32_t {
        success = 0
    };

    void gen_transpose_micro_4x4(uint32_t i_m,
                                 uint32_t i_n);
    void gen_transpose_micro_reminder(uint32_t i_m,
                                      uint32_t i_n);

    void gen_transpose(uint32_t i_m,
                       uint32_t i_n);

    void gen_zero(uint32_t m,
                  uint32_t n);

    void gen_relu(uint32_t m,
                  uint32_t n);

    void gen_identity(uint32_t m,
                      uint32_t n);

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