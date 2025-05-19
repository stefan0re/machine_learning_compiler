#include "Unary.h"

#include "../instructions/instructions.h"
#include "util.h"

namespace InstGen = mini_jit::instructions;

mini_jit::generator::Unary::error_t mini_jit::generator::Unary::generate(uint32_t m,
                                                                         uint32_t n,
                                                                         uint32_t trans_b,
                                                                         dtype_t dtype,
                                                                         ptype_t ptype) {
    // procedure call standard (store to stack)
    m_kernel.add_instr(0x6DBF27E8);
    m_kernel.add_instr(0x6DBF2FEA);
    m_kernel.add_instr(0x6DBF37EC);
    m_kernel.add_instr(0x6DBF3FEE);

    // procedure call standard (load from stack)
    m_kernel.add_instr(0x6CC13FEE);
    m_kernel.add_instr(0x6CC137EC);
    m_kernel.add_instr(0x6CC12FEA);
    m_kernel.add_instr(0x6CC127E8);

    //
    // load A
    // -------------------------------------------------------
    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v0,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v1,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v2,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v3,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v4,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v5,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v6,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_ld1_no_offset(
            InstGen::simd_fp_t::v7,
            Util::WORKING_ADDRESS_A_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_A_REG,
            Util::WORKING_ADDRESS_A_REG,
            4,
            /*no flags*/ 0));
    //
    // Transpose A to B
    // -------------------------------------------------------

    //
    // store B
    // -------------------------------------------------------
    //
    // load A
    // -------------------------------------------------------
    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v0,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v1,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v2,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v3,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v4,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v5,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v6,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    m_kernel.add_instr(
        InstGen::neon_st1_no_offset(
            InstGen::simd_fp_t::v7,
            Util::WORKING_ADDRESS_B_REG,
            InstGen::vector_count_t::vc4));
    m_kernel.add_instr(
        InstGen::base_add_imm(
            Util::WORKING_ADDRESS_B_REG,
            Util::WORKING_ADDRESS_B_REG,
            4,
            /*no flags*/ 0));

    // ret
    m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

    m_kernel.set_kernel();

    m_kernel.write("output_test_unary.bin");

    return mini_jit::generator::Unary::error_t::success;
}

mini_jit::generator::Unary::kernel_t mini_jit::generator::Unary::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
}