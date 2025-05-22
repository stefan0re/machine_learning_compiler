#include "Brgemm.h"
#include "util.h"
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

    BRGEMM_EXPECT((trans_a | trans_b | trans_c) == 0);
    BRGEMM_EXPECT(dtype == dtype_t::fp32);

    // procedure call standard (store to stack)
    m_kernel.add_instr(0x6DBF27E8);
    m_kernel.add_instr(0x6DBF2FEA);
    m_kernel.add_instr(0x6DBF37EC);
    m_kernel.add_instr(0x6DBF3FEE);

    /* Store pointers of A, B and C to x7, x8, x9 */
    m_kernel.add_instr(inst::InstGen::base_mov_register(Util::WORKING_ADDRESS_A_REG,
                                                        Util::INPUT_ADDRESS_A_REG));
    m_kernel.add_instr(inst::InstGen::base_mov_register(Util::WORKING_ADDRESS_B_REG,
                                                        Util::INPUT_ADDRESS_B_REG));
    m_kernel.add_instr(inst::InstGen::base_mov_register(Util::WORKING_ADDRESS_C_REG,
                                                        Util::INPUT_ADDRESS_C_REG));

    /* shift leading dimensions to 4 bytes  TODO!*/
    m_kernel.add_instr(0xd37ef463);
    m_kernel.add_instr(0xd37ef484);
    m_kernel.add_instr(0xd37ef4a5);

    Util::KernelSizes kernelsizes;
    Util::get_kernel_sizes(m, n, kernelsizes);



    Util::generator_load_reg_block(m_kernel, kernelsizes.kernel1, Util::WORKING_ADDRESS_C_REG);

    Util::generator_store_reg_block(m_kernel, kernelsizes.kernel1, Util::WORKING_ADDRESS_C_REG);


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