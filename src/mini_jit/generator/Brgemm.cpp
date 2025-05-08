#include "Brgemm.h"

#include "../instructions/instructions.h"

namespace inst = mini_jit::instructions;

mini_jit::generator::Brgemm::error_t mini_jit::generator::Brgemm::generate(uint32_t m,
                                                                           uint32_t n,
                                                                           uint32_t k,
                                                                           uint32_t br_size,
                                                                           uint32_t trans_a,
                                                                           uint32_t trans_b,
                                                                           uint32_t trans_c,
                                                                           dtype_t dtype) {
    // procedure call standard (store to stack)
    m_kernel.add_instr(0x6DBF27E8);
    m_kernel.add_instr(0x6DBF2FEA);
    m_kernel.add_instr(0x6DBF37EC);
    m_kernel.add_instr(0x6DBF3FEE);

    /* Store pointers of A, B and C to x7, x8, x9 */
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x7,
                                                        inst::InstGen::x0));
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x8,
                                                        inst::InstGen::x1));
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x9,
                                                        inst::InstGen::x2));

    /* TODO shift leading dimensions to 4 bytes */
    m_kernel.add_instr(0xd37ef463);
    m_kernel.add_instr(0xd37ef484);
    m_kernel.add_instr(0xd37ef4a5);

    // set register to reset B in the K loop:
    m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x15,
                                                   5,
                                                   0));
    m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x15,
                                                   inst::InstGen::x15,
                                                   inst::InstGen::x4));
    // set K loop register
    m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x10,
                                                   k,
                                                   0));

    for (size_t i = 0; i < 6; i++) {
        m_kernel.add_instr(inst::InstGen::neon_ld1_no_offset(static_cast<inst::InstGen::simd_fp_t>(4 * i),
                                                             inst::InstGen::x9,
                                                             inst::InstGen::vc4));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x9,
                                                                    inst::InstGen::x9,
                                                                    inst::InstGen::x5,
                                                                    0,
                                                                    0));
    }

    /* start k loop remember instruction count */
    size_t k_loop_count = m_kernel.get_size();
    // sub k loop counter
    m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x10,
                                                   inst::InstGen::x10,
                                                   1,
                                                   0));
    // load A TODO: ld1
    m_kernel.add_instr(0x4c402818);
    m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                inst::InstGen::x0,
                                                                inst::InstGen::x3,
                                                                0,
                                                                0));

    // load B
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(28 + i),
                                                   inst::InstGen::x1,
                                                   0));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                    inst::InstGen::x1,
                                                                    inst::InstGen::x4,
                                                                    0,
                                                                    0));
    }

    // issue fmla
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(0 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v28,
                                                               0));
    }

    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(4 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v29,
                                                               0));
    }
    // load rest of B

    m_kernel.add_instr(inst::InstGen::neon_ldr(inst::InstGen::v28,
                                               inst::InstGen::x1,
                                               0));
    m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                inst::InstGen::x1,
                                                                inst::InstGen::x4,
                                                                0,
                                                                0));
    m_kernel.add_instr(inst::InstGen::neon_ldr(inst::InstGen::v29,
                                               inst::InstGen::x1,
                                               0));

    // fmla
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(8 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v30,
                                                               0));
    }

    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(12 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v31,
                                                               0));
    }
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(16 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v28,
                                                               0));
    }
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_fmla_by_element(static_cast<inst::InstGen::simd_fp_t>(20 + i),
                                                               static_cast<inst::InstGen::simd_fp_t>(24 + i),
                                                               inst::InstGen::v29,
                                                               0));
    }

    /* set new B address */
    m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x1,
                                                                inst::InstGen::x1,
                                                                inst::InstGen::x15,
                                                                0,
                                                                0));
    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x1,
                                                   inst::InstGen::x1,
                                                   0x4,
                                                   0));

    /* cbnz K loop */
    m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x10,
                                                   (k_loop_count - m_kernel.get_size()) / 4 - 1));

    /* Store C */
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x9,
                                                        inst::InstGen::x2));

    for (size_t i = 0; i < 6; i++) {
        m_kernel.add_instr(inst::InstGen::neon_st1_no_offset(static_cast<inst::InstGen::simd_fp_t>(4 * i),
                                                             inst::InstGen::x9,
                                                             inst::InstGen::vc4));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x9,
                                                                    inst::InstGen::x9,
                                                                    inst::InstGen::x5,
                                                                    0,
                                                                    0));
    }

    // procedure call standard (load from stack)
    m_kernel.add_instr(0x6CC13FEE);
    m_kernel.add_instr(0x6CC137EC);
    m_kernel.add_instr(0x6CC12FEA);
    m_kernel.add_instr(0x6CC127E8);

    // ret
    m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

    m_kernel.set_kernel();

    m_kernel.write("output_test.bin");

    return mini_jit::generator::Brgemm::error_t::success;
}

mini_jit::generator::Brgemm::kernel_t mini_jit::generator::Brgemm::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
}