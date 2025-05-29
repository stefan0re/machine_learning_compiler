#ifndef TENGEN_TYPES_H
#define TENGEN_TYPES_H

#include <cstdint>

// Note: Implicit conversion to integers without the enum class keyword
// is used to allow for easier arithmetic operations and comparisons.

namespace TenGen::Types {
    //! general-purpose registers
    enum gpr_t : uint32_t {
        w0 = 0,
        w1 = 1,
        w2 = 2,
        w3 = 3,
        w4 = 4,
        w5 = 5,
        w6 = 6,
        w7 = 7,
        w8 = 8,
        w9 = 9,
        w10 = 10,
        w11 = 11,
        w12 = 12,
        w13 = 13,
        w14 = 14,
        w15 = 15,
        w16 = 16,
        w17 = 17,
        w18 = 18,
        w19 = 19,
        w20 = 20,
        w21 = 21,
        w22 = 22,
        w23 = 23,
        w24 = 24,
        w25 = 25,
        w26 = 26,
        w27 = 27,
        w28 = 28,
        w29 = 29,
        w30 = 30,

        x0 = 32 + 0,
        x1 = 32 + 1,
        x2 = 32 + 2,
        x3 = 32 + 3,
        x4 = 32 + 4,
        x5 = 32 + 5,
        x6 = 32 + 6,
        x7 = 32 + 7,
        x8 = 32 + 8,
        x9 = 32 + 9,
        x10 = 32 + 10,
        x11 = 32 + 11,
        x12 = 32 + 12,
        x13 = 32 + 13,
        x14 = 32 + 14,
        x15 = 32 + 15,
        x16 = 32 + 16,
        x17 = 32 + 17,
        x18 = 32 + 18,
        x19 = 32 + 19,
        x20 = 32 + 20,
        x21 = 32 + 21,
        x22 = 32 + 22,
        x23 = 32 + 23,
        x24 = 32 + 24,
        x25 = 32 + 25,
        x26 = 32 + 26,
        x27 = 32 + 27,
        x28 = 32 + 28,
        x29 = 32 + 29,
        x30 = 32 + 30,

        wzr = 31,
        xzr = 32 + 31,
        sp = 64 + 32 + 31
    };

    // Alias for general-purpose registers
    constexpr gpr_t INPUT_ADDRESS_A_REG = gpr_t::x0;
    constexpr gpr_t INPUT_ADDRESS_B_REG = gpr_t::x1;
    constexpr gpr_t INPUT_ADDRESS_C_REG = gpr_t::x2;

    constexpr gpr_t WORKING_ADDRESS_A_REG = gpr_t::x7;
    constexpr gpr_t WORKING_ADDRESS_B_REG = gpr_t::x8;
    constexpr gpr_t WORKING_ADDRESS_C_REG = gpr_t::x9;

    constexpr gpr_t LEADING_DIM_A_REG = gpr_t::x3;
    constexpr gpr_t LEADING_DIM_B_REG = gpr_t::x4;
    constexpr gpr_t LEADING_DIM_C_REG = gpr_t::x5;

    constexpr gpr_t K_LOOP_COUNT_REG = gpr_t::x10;
    constexpr gpr_t M_LOOP_COUNT_REG = gpr_t::x11;
    constexpr gpr_t N_LOOP_COUNT_REG = gpr_t::x12;
    constexpr gpr_t BR_LOOP_COUNT_REG = gpr_t::x16;

    constexpr gpr_t HELP_REG_1 = gpr_t::x13;
    constexpr gpr_t HELP_REG_2 = gpr_t::x14;
    constexpr gpr_t HELP_REG_3 = gpr_t::x15;

    //! simd&fp registers
    enum simd_fp_t : uint32_t {
        v0 = 0,
        v1 = 1,
        v2 = 2,
        v3 = 3,
        v4 = 4,
        v5 = 5,
        v6 = 6,
        v7 = 7,
        v8 = 8,
        v9 = 9,
        v10 = 10,
        v11 = 11,
        v12 = 12,
        v13 = 13,
        v14 = 14,
        v15 = 15,
        v16 = 16,
        v17 = 17,
        v18 = 18,
        v19 = 19,
        v20 = 20,
        v21 = 21,
        v22 = 22,
        v23 = 23,
        v24 = 24,
        v25 = 25,
        v26 = 26,
        v27 = 27,
        v28 = 28,
        v29 = 29,
        v30 = 30,
        v31 = 31
    };

    //! arrangement specifiers
    enum arr_spec_t : uint32_t {
        b = 0x0,
        h = 0x40000000,
        s = 0x80000000,
        d = 0xc0000000,
        q = 0x00800000,
    };

    enum element_spec_t : uint32_t {
        S2_0 = 0x00000000,
        S2_1 = 0x00200000,
        S2_2 = 0x00000800,
        S2_3 = 0x00200800,
        S4_0 = 0x40000000,
        S4_1 = 0x40200000,
        S4_2 = 0x40000800,
        S4_3 = 0x40200800,
        D2_0 = 0x40400000,
        D2_1 = 0x40400800
    };

    enum ld1_t : uint32_t {
        S2 = 0x800,
        S4 = 0x40000800,
        D2 = 0x40000c00
    };

    enum vector_count_t : uint32_t {
        vc1 = 0x5800,
        vc2 = 0x8800,
        vc3 = 0x4800,
        vc4 = 0x800,
    };

    enum ld1_opcode_t : uint32_t {
        one_regs = 0x7,
        two_regs = 0xa,
        three_regs = 0x6,
        four_regs = 0x2,
    };

    // data type
    enum class dtype_t : uint32_t {
        fp32 = 0,
        fp64 = 1
    };

    // primitive type
    enum class ptype_t : uint32_t {
        zero = 0,
        identity = 1,
        relu = 2
    };

    // error codes
    enum class error_t : int32_t {
        success = 0,
        bad_param = -1
    };

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

}  // namespace TenGen::Types

#endif  // TENGEN_TYPES_H
