#include "Brgemm.h"

#include <iostream>

#include "../instructions/instructions.h"
#include "util.h"

namespace inst = mini_jit::instructions;

void mini_jit::generator::Brgemm::gen_microkernel(backend::Kernel& i_kernel,
                                                  Util::KernelSize& i_kernelsize,
                                                  int32_t i_used_reg_count) {
    int32_t l_vreg_count = i_used_reg_count;
    int32_t l_n_count = 0;
    int32_t l_vreg_count_a = 0;

    int32_t l_m_block[3];
    l_m_block[0] = i_kernelsize.M / 4;
    l_m_block[1] = i_kernelsize.M % 4;

    // load values for A
    for (size_t i = 0; i < l_m_block[0]; i++) {
        m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(l_vreg_count),
                                                   Util::WORKING_ADDRESS_A_REG,
                                                   16,
                                                   inst::InstGen::arr_spec_t::q));
        l_vreg_count += 1;
        l_vreg_count_a++;
    }
    if (l_m_block[1] > 0) {
        // load remaining values for A
        if (l_m_block[1] == 1) {
            i_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(l_vreg_count),
                                                       Util::WORKING_ADDRESS_A_REG,
                                                       4,
                                                       inst::InstGen::arr_spec_t::s));

        } else if (l_m_block[1] == 2) {
            i_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(l_vreg_count),
                                                       Util::WORKING_ADDRESS_A_REG,
                                                       8,
                                                       inst::InstGen::arr_spec_t::d));
        } else if (l_m_block[1] == 3) {
            i_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(l_vreg_count),
                                                       Util::WORKING_ADDRESS_A_REG,
                                                       8,
                                                       inst::InstGen::arr_spec_t::d));

            m_kernel.add_instr(inst::InstGen::neon_ld1_scalar_index(static_cast<inst::InstGen::simd_fp_t>(l_vreg_count),
                                                                    Util::WORKING_ADDRESS_A_REG,
                                                                    2));
            i_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_A_REG,
                                                           Util::WORKING_ADDRESS_A_REG,
                                                           4,
                                                           0));
        }

        l_vreg_count += 1;
        l_vreg_count_a++;
    }

    // restore Working A
    m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_A_REG,
                                                   Util::WORKING_ADDRESS_A_REG,
                                                   l_m_block[0] * 16 + l_m_block[1] * 4,
                                                   0));

    int32_t l_b_vector_register = l_vreg_count;

    // compute with fmla
    for (size_t i = 0; i < i_used_reg_count; i++) {
        // load B value
        if (i % l_vreg_count_a == 0) {
            m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(l_b_vector_register),
                                                       Util::WORKING_ADDRESS_B_REG,
                                                       0,
                                                       inst::InstGen::arr_spec_t::s));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_B_REG,
                                                                        Util::WORKING_ADDRESS_B_REG,
                                                                        Util::LEADING_DIM_B_REG,
                                                                        0,
                                                                        0));
        }

        m_kernel.add_instr(inst::InstGen::neon_fmla_element(static_cast<inst::InstGen::simd_fp_t>(i),
                                                            static_cast<inst::InstGen::simd_fp_t>(i_used_reg_count + (i % l_vreg_count_a)),
                                                            static_cast<inst::InstGen::simd_fp_t>(l_b_vector_register),
                                                            inst::InstGen::element_spec_t::S4_0));
    }

    // restore Working B
    m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::HELP_REG_1,
                                                   i_kernelsize.N,
                                                   0));
    m_kernel.add_instr(inst::InstGen::base_mul_reg(Util::HELP_REG_1,
                                                   Util::HELP_REG_1,
                                                   Util::LEADING_DIM_B_REG));
    m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(Util::WORKING_ADDRESS_B_REG,
                                                                Util::WORKING_ADDRESS_B_REG,
                                                                Util::HELP_REG_1,
                                                                0,
                                                                0));
}

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

    Util::KernelSize kernelsize_big;
    Util::KernelSize kernelsize_reminder_big;
    Util::KernelSize kernelsize_small;
    Util::KernelSize kernelsize_reminder_small;
    int reg_count_big = 0;
    int reg_count_small = 0;
    int reg_count_reminder_big = 0;
    int reg_count_reminder_small = 0;
    Util::get_kernel_sizes_brgemm(m, n, kernelsize_big, kernelsize_small, reg_count_big, reg_count_small);

    int full_m_loop = m / kernelsize_big.M;
    int rem_m_loop = m % kernelsize_big.M;

    int full_n_loop = n / kernelsize_big.N;
    int rem_n_loop = n % kernelsize_big.N;

    kernelsize_reminder_big.M = rem_m_loop;
    kernelsize_reminder_big.N = kernelsize_big.N;

    kernelsize_reminder_small.M = rem_m_loop;
    kernelsize_reminder_small.N = kernelsize_small.N;

    if (full_n_loop > 0) {
        // set N loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::N_LOOP_COUNT_REG, full_n_loop, 0));
        // sub N loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::N_LOOP_COUNT_REG,
                                                       Util::N_LOOP_COUNT_REG,
                                                       1,
                                                       0));

        // get N loop position
        std::size_t n_loop_pos = m_kernel.get_size();

        // set M loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::M_LOOP_COUNT_REG, full_m_loop, 0));
        // sub M loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::M_LOOP_COUNT_REG,
                                                       Util::M_LOOP_COUNT_REG,
                                                       1,
                                                       0));
        // get M loop position
        std::size_t m_loop_pos = m_kernel.get_size();

        Util::generator_load_reg_block(m_kernel, kernelsize_big, Util::WORKING_ADDRESS_C_REG);

        // set K loop  counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::K_LOOP_COUNT_REG, k, 0));
        // sub K loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::K_LOOP_COUNT_REG,
                                                       Util::K_LOOP_COUNT_REG,
                                                       1,
                                                       0));

        // get k loop position
        std::size_t k_loop_pos = m_kernel.get_size();

        mini_jit::generator::Brgemm::gen_microkernel(m_kernel, kernelsize_big, reg_count_big);

        // adjust Working A and B
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                    Util::WORKING_ADDRESS_A_REG,
                                                                    Util::LEADING_DIM_A_REG,
                                                                    0,
                                                                    0));
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG,
                                                       Util::WORKING_ADDRESS_B_REG,
                                                       4,
                                                       0));

        /* cbnz K loop */
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::K_LOOP_COUNT_REG,
                                                       (k_loop_pos - m_kernel.get_size()) / 4 - 1));

        Util::generator_store_reg_block(m_kernel, kernelsize_big, Util::WORKING_ADDRESS_C_REG);

        // restore Working A
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::HELP_REG_2, k, 0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(Util::HELP_REG_2,
                                                       Util::HELP_REG_2,
                                                       Util::LEADING_DIM_A_REG));
        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                    Util::WORKING_ADDRESS_A_REG,
                                                                    Util::HELP_REG_2,
                                                                    0,
                                                                    0));
        // Adjust A to next M block
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_A_REG,
                                                       Util::WORKING_ADDRESS_A_REG,
                                                       kernelsize_big.M * 4,
                                                       0));
        // restore Working B
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_B_REG,
                                                       Util::WORKING_ADDRESS_B_REG,
                                                       k * 4,
                                                       0));
        // restore Working C
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_C_REG,
                                                       Util::WORKING_ADDRESS_C_REG,
                                                       kernelsize_big.M * 4,
                                                       0));

        /* cbnz M loop */
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::M_LOOP_COUNT_REG,
                                                       (m_loop_pos - m_kernel.get_size()) / 4 - 1));

        /****************************/
        // compute reminder block
        /****************************/
        if (rem_m_loop > 0) {
            Util::generator_load_reg_block(m_kernel, kernelsize_reminder_big, Util::WORKING_ADDRESS_C_REG);

            // set K loop  counter
            m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::K_LOOP_COUNT_REG, k, 0));
            // sub K loop register
            m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::K_LOOP_COUNT_REG,
                                                           Util::K_LOOP_COUNT_REG,
                                                           1,
                                                           0));

            // get k loop position
            k_loop_pos = m_kernel.get_size();

            reg_count_reminder_big = ((kernelsize_reminder_big.M + 3) / 4) * kernelsize_reminder_big.N;

            mini_jit::generator::Brgemm::gen_microkernel(m_kernel, kernelsize_reminder_big, reg_count_reminder_big);

            // adjust Working A and B
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                        Util::WORKING_ADDRESS_A_REG,
                                                                        Util::LEADING_DIM_A_REG,
                                                                        0,
                                                                        0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG,
                                                           Util::WORKING_ADDRESS_B_REG,
                                                           4,
                                                           0));

            /* cbnz K loop */
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::K_LOOP_COUNT_REG,
                                                           (k_loop_pos - m_kernel.get_size()) / 4 - 1));

            Util::generator_store_reg_block(m_kernel, kernelsize_reminder_big, Util::WORKING_ADDRESS_C_REG);
        }

        // restore Working A
        m_kernel.add_instr(inst::InstGen::base_mov_register(Util::WORKING_ADDRESS_A_REG,
                                                            Util::INPUT_ADDRESS_A_REG));

        // restore Working B
        if (rem_m_loop > 0) {
            m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_B_REG,
                                                           Util::WORKING_ADDRESS_B_REG,
                                                           k * 4,
                                                           0));
        }
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::HELP_REG_1,
                                                       kernelsize_big.N,
                                                       0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(Util::HELP_REG_1,
                                                       Util::HELP_REG_1,
                                                       Util::LEADING_DIM_B_REG));

        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_B_REG,
                                                                    Util::WORKING_ADDRESS_B_REG,
                                                                    Util::HELP_REG_1,
                                                                    0,
                                                                    0));
        // restore Working C
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_C_REG,
                                                       Util::WORKING_ADDRESS_C_REG,
                                                       kernelsize_big.M * 4 * full_m_loop,
                                                       0));
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::HELP_REG_3,
                                                       kernelsize_big.N,
                                                       0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(Util::HELP_REG_3,
                                                       Util::HELP_REG_3,
                                                       Util::LEADING_DIM_C_REG));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_C_REG,
                                                                    Util::WORKING_ADDRESS_C_REG,
                                                                    Util::HELP_REG_3,
                                                                    0,
                                                                    0));

        // cbnz N loop
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::N_LOOP_COUNT_REG,
                                                       (n_loop_pos - m_kernel.get_size()) / 4 - 1));
    }
    if (rem_n_loop > 0) {
        // set M loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::M_LOOP_COUNT_REG, full_m_loop, 0));
        // sub M loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::M_LOOP_COUNT_REG,
                                                       Util::M_LOOP_COUNT_REG,
                                                       1,
                                                       0));
        // get M loop position
        std::size_t m_loop_pos = m_kernel.get_size();

        Util::generator_load_reg_block(m_kernel, kernelsize_small, Util::WORKING_ADDRESS_C_REG);
        // set K loop  counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::K_LOOP_COUNT_REG, k, 0));
        // sub K loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::K_LOOP_COUNT_REG,
                                                       Util::K_LOOP_COUNT_REG,
                                                       1,
                                                       0));
        // get k loop position
        std::size_t k_loop_pos = m_kernel.get_size();

        mini_jit::generator::Brgemm::gen_microkernel(m_kernel, kernelsize_small, reg_count_small);

        // adjust Working A and B
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                    Util::WORKING_ADDRESS_A_REG,
                                                                    Util::LEADING_DIM_A_REG,
                                                                    0,
                                                                    0));
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG,
                                                       Util::WORKING_ADDRESS_B_REG,
                                                       4,
                                                       0));
        /* cbnz K loop */
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::K_LOOP_COUNT_REG,
                                                       (k_loop_pos - m_kernel.get_size()) / 4 - 1));

        Util::generator_store_reg_block(m_kernel, kernelsize_small, Util::WORKING_ADDRESS_C_REG);

        // restore Working A
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::HELP_REG_2, k, 0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(Util::HELP_REG_2,
                                                       Util::HELP_REG_2,
                                                       Util::LEADING_DIM_A_REG));
        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                    Util::WORKING_ADDRESS_A_REG,
                                                                    Util::HELP_REG_2,
                                                                    0,
                                                                    0));
        // Adjust A to next M block
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_A_REG,
                                                       Util::WORKING_ADDRESS_A_REG,
                                                       kernelsize_small.M * 4,
                                                       0));
        // restore Working B
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_B_REG,
                                                       Util::WORKING_ADDRESS_B_REG,
                                                       k * 4,
                                                       0));
        // restore Working C
        m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_C_REG,
                                                       Util::WORKING_ADDRESS_C_REG,
                                                       kernelsize_small.M * 4,
                                                       0));
        /* cbnz M loop */
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::M_LOOP_COUNT_REG,
                                                       (m_loop_pos - m_kernel.get_size()) / 4 - 1));

        if (rem_m_loop > 0) {
            Util::generator_load_reg_block(m_kernel, kernelsize_reminder_small, Util::WORKING_ADDRESS_C_REG);

            // set K loop  counter
            m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::K_LOOP_COUNT_REG, k, 0));
            // sub K loop register
            m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::K_LOOP_COUNT_REG,
                                                           Util::K_LOOP_COUNT_REG,
                                                           1,
                                                           0));

            // get k loop position
            k_loop_pos = m_kernel.get_size();

            reg_count_reminder_small = ((kernelsize_reminder_small.M + 3) / 4) * kernelsize_reminder_small.N;

            mini_jit::generator::Brgemm::gen_microkernel(m_kernel, kernelsize_reminder_small, reg_count_reminder_small);

            // adjust Working A and B
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(Util::WORKING_ADDRESS_A_REG,
                                                                        Util::WORKING_ADDRESS_A_REG,
                                                                        Util::LEADING_DIM_A_REG,
                                                                        0,
                                                                        0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG,
                                                           Util::WORKING_ADDRESS_B_REG,
                                                           4,
                                                           0));

            /* cbnz K loop */
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::K_LOOP_COUNT_REG,
                                                           (k_loop_pos - m_kernel.get_size()) / 4 - 1));

            Util::generator_store_reg_block(m_kernel, kernelsize_reminder_small, Util::WORKING_ADDRESS_C_REG);
        }
        }

    // procedure call standard (load from stack)
    m_kernel.add_instr(0x6CC13FEE);
    m_kernel.add_instr(0x6CC137EC);
    m_kernel.add_instr(0x6CC12FEA);
    m_kernel.add_instr(0x6CC127E8);

    // print out all 4 kernel types:
    std::cout << "Kernel sizes:" << std::endl;
    std::cout << "  Big kernel: M = " << kernelsize_big.M << ", N = " << kernelsize_big.N << std::endl;
    std::cout << "  Reminder big kernel: M = " << kernelsize_reminder_big.M << ", N = " << kernelsize_reminder_big.N << std::endl;
    std::cout << "  Small kernel: M = " << kernelsize_small.M << ", N = " << kernelsize_small.N << std::endl;
    std::cout << "  Reminder small kernel: M = " << kernelsize_reminder_small.M << ", N = " << kernelsize_reminder_small.N << std::endl;

    // ret
    m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

    m_kernel.set_kernel();

    m_kernel.write("output_test.bin");

    return mini_jit::generator::Brgemm::error_t::success;
}

mini_jit::generator::Brgemm::kernel_t mini_jit::generator::Brgemm::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
}