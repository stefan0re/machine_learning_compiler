#include "Brgemm.h"

#include <iostream>

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
    // m_kernel.add_instr(0);
    // procedure call standard (store to stack)
    /*
    m_kernel.add_instr(0xA9BF53F3);
    m_kernel.add_instr(0xA9BF5BF5);
    m_kernel.add_instr(0xA9BF63F7);
    m_kernel.add_instr(0xA9BF6BF9);
    m_kernel.add_instr(0xA9BF73FB);
    m_kernel.add_instr(0xA9BF7BFD);

    m_kernel.add_instr(0x6DBF27E8);
    m_kernel.add_instr(0x6DBF2FEA);
    m_kernel.add_instr(0x6DBF37EC);
    m_kernel.add_instr(0x6DBF3FEE);
    */
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
                                                   5));
    m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x15,
                                                   inst::InstGen::x15,
                                                   inst::InstGen::x4));
    // set K loop register
    m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x10,
                                                   k));
    // load C TODO: loop and real instructions
    // row 1
    m_kernel.add_instr(0x4c402920);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));
    m_kernel.add_instr(0x4c402924);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));

    m_kernel.add_instr(0x4c402928);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));

    m_kernel.add_instr(0x4c40292c);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));

    m_kernel.add_instr(0x4c402930);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));
    // row 6
    m_kernel.add_instr(0x4c402934);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x9,
                                                       inst::InstGen::x9,
                                                       inst::InstGen::x5,
                                                       0,
                                                       0));

    /* start k loop remember instruction count */

    size_t k_loop_count = m_kernel.get_size();
    std::cout << k_loop_count << std::endl;
    // sub k loop counter
    m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x10,
                                                   inst::InstGen::x10,
                                                   1,
                                                   0));
    // load A TODO: ld1
    m_kernel.add_instr(0x4c402818);
    m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x0,
                                                       inst::InstGen::x0,
                                                       inst::InstGen::x3,
                                                       0,
                                                       0));

    // load B
    for (size_t i = 0; i < 4; i++) {
        m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(28 + i),
                                                   inst::InstGen::x1,
                                                   0));
        m_kernel.add_instr(inst::InstGen::base_add_shifted(inst::InstGen::x1,
                                                           inst::InstGen::x1,
                                                           inst::InstGen::x4,
                                                           0,
                                                           0));
    }

    /**
     * @todo: FMLA by element
     *        ld1
     *        st1
     *        check cbnz
     */

    // procedure call standard (load from stack)
    /*
    m_kernel.add_instr(0x6CC13FEE);
    m_kernel.add_instr(0x6CC137EC);
    m_kernel.add_instr(0x6CC12FEA);
    m_kernel.add_instr(0x6CC127E8);

    m_kernel.add_instr(0xA8C17BFD);
    m_kernel.add_instr(0xA8C173FB);
    m_kernel.add_instr(0xA8C16BF9);
    m_kernel.add_instr(0xA8C163F7);
    m_kernel.add_instr(0xA8C15BF5);
    m_kernel.add_instr(0xA8C153F3);
    */
    // ret
    m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

    m_kernel.set_kernel();

    m_kernel.write("output_test.bin");

    return mini_jit::generator::Brgemm::error_t::success;
}

mini_jit::generator::Brgemm::kernel_t mini_jit::generator::Brgemm::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
}