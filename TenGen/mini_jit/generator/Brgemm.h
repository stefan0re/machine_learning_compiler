#ifndef TENGEN_MINI_JIT_GENERATOR_BRGEMM_H
#define TENGEN_MINI_JIT_GENERATOR_BRGEMM_H

#include <cstdint>

#include "TenGen.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using namespace TenGen::MiniJit::Instructions::Encoding;
using Kernel = TenGen::MiniJit::Backend::Kernel;
using Util = TenGen::MiniJit::Generator::Util;

#define BRGEMM_EXPECT(cond)                     \
    do {                                        \
        if (!(cond)) return error_t::bad_param; \
    } while (0)

namespace TenGen::MiniJit::Generator {
    class Brgemm {
       private:
        //! kernel backend
        Kernel m_kernel;

        /**
         * @brief Generate a kernel for batch-reduce matrix multiplication.
         * @param m number of rows in A and C.
         * @param n number of columns in B and C.
         * @param k number of columns in A and rows in B.
         * @param br_size batch-reduce size.
         * @param trans_a 0 if A is stored in column-major order, 1 if A is stored in row-major order.
         * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
         * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
         * @param dtype data type of the matrices.
         * @return error_t::success on success, another error_t value otherwise.
         **/
        TenGen::Types::error_t generate(uint32_t m,
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
            m_kernel.add_instr(base_mov_register(WORKING_ADDRESS_A_REG,
                                                 INPUT_ADDRESS_A_REG));
            m_kernel.add_instr(base_mov_register(WORKING_ADDRESS_B_REG,
                                                 INPUT_ADDRESS_B_REG));
            m_kernel.add_instr(base_mov_register(WORKING_ADDRESS_C_REG,
                                                 INPUT_ADDRESS_C_REG));

            /* shift leading dimensions to 4 bytes  TODO!*/
            m_kernel.add_instr(0xd37ef463);
            m_kernel.add_instr(0xd37ef484);
            m_kernel.add_instr(0xd37ef4a5);

            KernelSize kernelsize_big;
            KernelSize kernelsize_reminder_big;
            KernelSize kernelsize_small;
            KernelSize kernelsize_reminder_small;
            int reg_count_big = 0;
            int reg_count_small = 0;
            int reg_count_reminder_big = 0;
            int reg_count_reminder_small = 0;
            std::size_t br_loop_pos = 0;
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
                m_kernel.add_instr(base_mov_imm(N_LOOP_COUNT_REG, full_n_loop, 0));
                // sub N loop register
                m_kernel.add_instr(base_sub_imm(N_LOOP_COUNT_REG,
                                                N_LOOP_COUNT_REG,
                                                1,
                                                0));

                // get N loop position
                std::size_t n_loop_pos = m_kernel.get_size();

                // set M loop counter
                m_kernel.add_instr(base_mov_imm(M_LOOP_COUNT_REG, full_m_loop, 0));
                // sub M loop register
                m_kernel.add_instr(base_sub_imm(M_LOOP_COUNT_REG,
                                                M_LOOP_COUNT_REG,
                                                1,
                                                0));
                // get M loop position
                std::size_t m_loop_pos = m_kernel.get_size();

                Util::generator_load_reg_block(m_kernel, kernelsize_big, WORKING_ADDRESS_C_REG);

                if (br_size > 1) {
                    // set BR loop counter
                    m_kernel.add_instr(base_mov_imm(BR_LOOP_COUNT_REG, br_size, 0));
                    // sub BR loop register
                    m_kernel.add_instr(base_sub_imm(BR_LOOP_COUNT_REG,
                                                    BR_LOOP_COUNT_REG,
                                                    1,
                                                    0));
                    // get BR loop position
                    br_loop_pos = m_kernel.get_size();
                }
                // set K loop  counter
                m_kernel.add_instr(base_mov_imm(K_LOOP_COUNT_REG, k, 0));
                // sub K loop register
                m_kernel.add_instr(base_sub_imm(K_LOOP_COUNT_REG,
                                                K_LOOP_COUNT_REG,
                                                1,
                                                0));

                // get k loop position
                std::size_t k_loop_pos = m_kernel.get_size();

                Brgemm::gen_microkernel(m_kernel, kernelsize_big, reg_count_big);

                // adjust Working A and B
                m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_A_REG,
                                                             WORKING_ADDRESS_A_REG,
                                                             LEADING_DIM_A_REG,
                                                             0,
                                                             0));
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG,
                                                WORKING_ADDRESS_B_REG,
                                                4,
                                                0));

                /* cbnz K loop */
                m_kernel.add_instr(base_br_cbnz(K_LOOP_COUNT_REG,
                                                (k_loop_pos - m_kernel.get_size()) / 4 - 1));

                if (br_size > 1) {
                    // adjust Working A
                    // nothing to do because A can continue perfectly

                    // adjust Working B
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    k * 4,
                                                    0));
                    // add BR stride to Working B
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1, n, 0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_B_REG));
                    m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                                 WORKING_ADDRESS_B_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));

                    // cbnz BR loop
                    m_kernel.add_instr(base_br_cbnz(BR_LOOP_COUNT_REG,
                                                    (br_loop_pos - m_kernel.get_size()) / 4 - 1));

                    // restore Working A
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                    br_size,
                                                    0));
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                    k,
                                                    0));

                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    HELP_REG_2));

                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_A_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));
                    // restore Working B
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                    br_size,
                                                    0));
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                    n,
                                                    0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    HELP_REG_2));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_B_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_B_REG,
                                                                 WORKING_ADDRESS_B_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));
                }

                Util::generator_store_reg_block(m_kernel, kernelsize_big, WORKING_ADDRESS_C_REG);

                // restore Working A
                if (br_size < 2) {
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2, k, 0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_2,
                                                    HELP_REG_2,
                                                    LEADING_DIM_A_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 HELP_REG_2,
                                                                 0,
                                                                 0));
                }
                // Adjust A to next M block
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_A_REG,
                                                WORKING_ADDRESS_A_REG,
                                                kernelsize_big.M * 4,
                                                0));
                if (br_size < 2) {
                    // restore Working B
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    k * 4,
                                                    0));
                }
                // restore Working C
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_C_REG,
                                                WORKING_ADDRESS_C_REG,
                                                kernelsize_big.M * 4,
                                                0));

                /* cbnz M loop */
                m_kernel.add_instr(base_br_cbnz(M_LOOP_COUNT_REG,
                                                (m_loop_pos - m_kernel.get_size()) / 4 - 1));

                /****************************/
                // compute M reminder block
                /****************************/
                if (rem_m_loop > 0) {
                    Util::generator_load_reg_block(m_kernel, kernelsize_reminder_big, WORKING_ADDRESS_C_REG);

                    if (br_size > 1) {
                        // set BR loop counter
                        m_kernel.add_instr(base_mov_imm(BR_LOOP_COUNT_REG, br_size, 0));
                        // sub BR loop register
                        m_kernel.add_instr(base_sub_imm(BR_LOOP_COUNT_REG,
                                                        BR_LOOP_COUNT_REG,
                                                        1,
                                                        0));
                        // get BR loop position
                        br_loop_pos = m_kernel.get_size();
                    }

                    // set K loop  counter
                    m_kernel.add_instr(base_mov_imm(K_LOOP_COUNT_REG, k, 0));
                    // sub K loop register
                    m_kernel.add_instr(base_sub_imm(K_LOOP_COUNT_REG,
                                                    K_LOOP_COUNT_REG,
                                                    1,
                                                    0));

                    // get k loop position
                    k_loop_pos = m_kernel.get_size();

                    reg_count_reminder_big = ((kernelsize_reminder_big.M + 3) / 4) * kernelsize_reminder_big.N;

                    Brgemm::gen_microkernel(m_kernel, kernelsize_reminder_big, reg_count_reminder_big);

                    // adjust Working A and B
                    m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 LEADING_DIM_A_REG,
                                                                 0,
                                                                 0));
                    m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    4,
                                                    0));

                    /* cbnz K loop */
                    m_kernel.add_instr(base_br_cbnz(K_LOOP_COUNT_REG,
                                                    (k_loop_pos - m_kernel.get_size()) / 4 - 1));

                    if (br_size > 1) {
                        // adjust Working A
                        // nothing to do because A can continue perfectly

                        // adjust Working B
                        m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                        WORKING_ADDRESS_B_REG,
                                                        k * 4,
                                                        0));
                        // add BR stride to Working B
                        m_kernel.add_instr(base_mov_imm(HELP_REG_1, n, 0));
                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        LEADING_DIM_B_REG));
                        m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                                     WORKING_ADDRESS_B_REG,
                                                                     HELP_REG_1,
                                                                     0,
                                                                     0));

                        // cbnz BR loop
                        m_kernel.add_instr(base_br_cbnz(BR_LOOP_COUNT_REG,
                                                        (br_loop_pos - m_kernel.get_size()) / 4 - 1));

                        // restore Working A
                        m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                        br_size,
                                                        0));
                        m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                        k,
                                                        0));

                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        HELP_REG_2));

                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        LEADING_DIM_A_REG));
                        m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_A_REG,
                                                                     WORKING_ADDRESS_A_REG,
                                                                     HELP_REG_1,
                                                                     0,
                                                                     0));
                        // restore Working B
                        m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                        br_size,
                                                        0));
                        m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                        n,
                                                        0));
                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        HELP_REG_2));
                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        LEADING_DIM_B_REG));
                        m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_B_REG,
                                                                     WORKING_ADDRESS_B_REG,
                                                                     HELP_REG_1,
                                                                     0,
                                                                     0));
                    }

                    Util::generator_store_reg_block(m_kernel, kernelsize_reminder_big, WORKING_ADDRESS_C_REG);
                }

                // restore Working A
                m_kernel.add_instr(base_mov_register(WORKING_ADDRESS_A_REG,
                                                     INPUT_ADDRESS_A_REG));

                // restore Working B
                if (rem_m_loop > 0 && br_size < 2) {
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    k * 4,
                                                    0));
                }
                m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                kernelsize_big.N,
                                                0));
                m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                HELP_REG_1,
                                                LEADING_DIM_B_REG));

                m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                             WORKING_ADDRESS_B_REG,
                                                             HELP_REG_1,
                                                             0,
                                                             0));
                // restore Working C
                m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_C_REG,
                                                WORKING_ADDRESS_C_REG,
                                                kernelsize_big.M * 4 * full_m_loop,
                                                0));
                m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                kernelsize_big.N,
                                                0));
                m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                HELP_REG_1,
                                                LEADING_DIM_C_REG));
                m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_C_REG,
                                                             WORKING_ADDRESS_C_REG,
                                                             HELP_REG_1,
                                                             0,
                                                             0));

                // cbnz N loop
                m_kernel.add_instr(base_br_cbnz(N_LOOP_COUNT_REG,
                                                (n_loop_pos - m_kernel.get_size()) / 4 - 1));
            }

            /****************************/
            // compute N reminder Block
            /****************************/
            if (rem_n_loop > 0) {
                // set M loop counter
                m_kernel.add_instr(base_mov_imm(M_LOOP_COUNT_REG, full_m_loop, 0));
                // sub M loop register
                m_kernel.add_instr(base_sub_imm(M_LOOP_COUNT_REG,
                                                M_LOOP_COUNT_REG,
                                                1,
                                                0));
                // get M loop position
                std::size_t m_loop_pos = m_kernel.get_size();

                Util::generator_load_reg_block(m_kernel, kernelsize_small, WORKING_ADDRESS_C_REG);
                if (br_size > 1) {
                    // set BR loop counter
                    m_kernel.add_instr(base_mov_imm(BR_LOOP_COUNT_REG, br_size, 0));
                    // sub BR loop register
                    m_kernel.add_instr(base_sub_imm(BR_LOOP_COUNT_REG,
                                                    BR_LOOP_COUNT_REG,
                                                    1,
                                                    0));
                    // get BR loop position
                    br_loop_pos = m_kernel.get_size();
                }
                // set K loop  counter
                m_kernel.add_instr(base_mov_imm(K_LOOP_COUNT_REG, k, 0));
                // sub K loop register
                m_kernel.add_instr(base_sub_imm(K_LOOP_COUNT_REG,
                                                K_LOOP_COUNT_REG,
                                                1,
                                                0));
                // get k loop position
                std::size_t k_loop_pos = m_kernel.get_size();

                Brgemm::gen_microkernel(m_kernel, kernelsize_small, reg_count_small);

                // adjust Working A and B
                m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_A_REG,
                                                             WORKING_ADDRESS_A_REG,
                                                             LEADING_DIM_A_REG,
                                                             0,
                                                             0));
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG,
                                                WORKING_ADDRESS_B_REG,
                                                4,
                                                0));
                /* cbnz K loop */
                m_kernel.add_instr(base_br_cbnz(K_LOOP_COUNT_REG,
                                                (k_loop_pos - m_kernel.get_size()) / 4 - 1));

                if (br_size > 1) {
                    // adjust Working A
                    // nothing to do because A can continue perfectly

                    // adjust Working B
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    k * 4,
                                                    0));
                    // add BR stride to Working B
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1, n, 0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_B_REG));
                    m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                                 WORKING_ADDRESS_B_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));

                    // cbnz BR loop
                    m_kernel.add_instr(base_br_cbnz(BR_LOOP_COUNT_REG,
                                                    (br_loop_pos - m_kernel.get_size()) / 4 - 1));

                    // restore Working A
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                    br_size,
                                                    0));
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                    k,
                                                    0));

                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    HELP_REG_2));

                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_A_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));
                    // restore Working B
                    m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                                    br_size,
                                                    0));
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2,
                                                    n,
                                                    0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    HELP_REG_2));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                    HELP_REG_1,
                                                    LEADING_DIM_B_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_B_REG,
                                                                 WORKING_ADDRESS_B_REG,
                                                                 HELP_REG_1,
                                                                 0,
                                                                 0));
                }

                Util::generator_store_reg_block(m_kernel, kernelsize_small, WORKING_ADDRESS_C_REG);

                // restore Working A
                if (br_size < 2) {
                    m_kernel.add_instr(base_mov_imm(HELP_REG_2, k, 0));
                    m_kernel.add_instr(base_mul_reg(HELP_REG_2,
                                                    HELP_REG_2,
                                                    LEADING_DIM_A_REG));
                    m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 HELP_REG_2,
                                                                 0,
                                                                 0));
                }

                // Adjust A to next M block
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_A_REG,
                                                WORKING_ADDRESS_A_REG,
                                                kernelsize_small.M * 4,
                                                0));
                // restore Working B
                if (br_size < 2) {
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    k * 4,
                                                    0));
                }
                // restore Working C
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_C_REG,
                                                WORKING_ADDRESS_C_REG,
                                                kernelsize_small.M * 4,
                                                0));
                /* cbnz M loop */
                m_kernel.add_instr(base_br_cbnz(M_LOOP_COUNT_REG,
                                                (m_loop_pos - m_kernel.get_size()) / 4 - 1));

                if (rem_m_loop > 0) {
                    Util::generator_load_reg_block(m_kernel, kernelsize_reminder_small, WORKING_ADDRESS_C_REG);

                    if (br_size > 1) {
                        // set BR loop counter
                        m_kernel.add_instr(base_mov_imm(BR_LOOP_COUNT_REG, br_size, 0));
                        // sub BR loop register
                        m_kernel.add_instr(base_sub_imm(BR_LOOP_COUNT_REG,
                                                        BR_LOOP_COUNT_REG,
                                                        1,
                                                        0));
                        // get BR loop position
                        br_loop_pos = m_kernel.get_size();
                    }

                    // set K loop  counter
                    m_kernel.add_instr(base_mov_imm(K_LOOP_COUNT_REG, k, 0));
                    // sub K loop register
                    m_kernel.add_instr(base_sub_imm(K_LOOP_COUNT_REG,
                                                    K_LOOP_COUNT_REG,
                                                    1,
                                                    0));

                    // get k loop position
                    k_loop_pos = m_kernel.get_size();

                    reg_count_reminder_small = ((kernelsize_reminder_small.M + 3) / 4) * kernelsize_reminder_small.N;

                    Brgemm::gen_microkernel(m_kernel, kernelsize_reminder_small, reg_count_reminder_small);

                    // adjust Working A and B
                    m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_A_REG,
                                                                 WORKING_ADDRESS_A_REG,
                                                                 LEADING_DIM_A_REG,
                                                                 0,
                                                                 0));
                    m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG,
                                                    WORKING_ADDRESS_B_REG,
                                                    4,
                                                    0));

                    /* cbnz K loop */
                    m_kernel.add_instr(base_br_cbnz(K_LOOP_COUNT_REG,
                                                    (k_loop_pos - m_kernel.get_size()) / 4 - 1));

                    if (br_size > 1) {
                        // adjust Working A
                        // nothing to do because A can continue perfectly

                        // adjust Working B
                        m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG,
                                                        WORKING_ADDRESS_B_REG,
                                                        k * 4,
                                                        0));
                        // add BR stride to Working B
                        m_kernel.add_instr(base_mov_imm(HELP_REG_1, n, 0));
                        m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                                        HELP_REG_1,
                                                        LEADING_DIM_B_REG));
                        m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                                     WORKING_ADDRESS_B_REG,
                                                                     HELP_REG_1,
                                                                     0,
                                                                     0));

                        // cbnz BR loop
                        m_kernel.add_instr(base_br_cbnz(BR_LOOP_COUNT_REG,
                                                        (br_loop_pos - m_kernel.get_size()) / 4 - 1));
                    }

                    Util::generator_store_reg_block(m_kernel, kernelsize_reminder_small, WORKING_ADDRESS_C_REG);
                }
            }

            // procedure call standard (load from stack)
            m_kernel.add_instr(0x6CC13FEE);
            m_kernel.add_instr(0x6CC137EC);
            m_kernel.add_instr(0x6CC12FEA);
            m_kernel.add_instr(0x6CC127E8);

            // ret
            m_kernel.add_instr(base_ret());

            m_kernel.set_kernel();

            m_kernel.write("output_test.bin");

            return error_t::success;
        }

        /*
         * Kernel type.
         * The kernel is a function that takes the following parameters:
         * - a: pointer to first column-major A matrix.
         * - b: pointer to first column-major B matrix.
         * - c: pointer to first column-major C matrix.
         * - lda: leading dimension of A.
         * - ldb: leading dimension of B.
         * - ldc: leading dimension of C.
         * - br_stride_a: stride between two A matrices (in elements, not bytes).
         * - br_stride_b: stride between two B matrices (in elements, not bytes).
         */
        using kernel_t = void (*)(void const* a,
                                  void const* b,
                                  void* c,
                                  int64_t lda,
                                  int64_t ldb,
                                  int64_t ldc,
                                  int64_t br_stride_a,
                                  int64_t br_stride_b);

        /**
         * @brief Get the generated kernel: C += sum_i(A_i * B_i).
         * @return pointer to the generated kernel.
         **/
        kernel_t get_kernel() const {
            return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
        }

        /**
         * @brief Generate the inner microkernel of the matrix multplication
         *
         */
        void gen_microkernel(Kernel& i_kernel,
                             KernelSize& i_kernelsize,
                             int32_t i_used_reg_count) {
            int32_t l_vreg_count = i_used_reg_count;
            int32_t l_n_count = 0;
            int32_t l_vreg_count_a = 0;

            int32_t l_m_block[3];
            l_m_block[0] = i_kernelsize.M / 4;
            l_m_block[1] = i_kernelsize.M % 4;

            // load values for A
            for (size_t i = 0; i < l_m_block[0]; i++) {
                m_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_vreg_count),
                                            WORKING_ADDRESS_A_REG,
                                            16,
                                            arr_spec_t::q));
                l_vreg_count += 1;
                l_vreg_count_a++;
            }
            if (l_m_block[1] > 0) {
                // load remaining values for A
                if (l_m_block[1] == 1) {
                    i_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_vreg_count),
                                                WORKING_ADDRESS_A_REG,
                                                4,
                                                arr_spec_t::s));

                } else if (l_m_block[1] == 2) {
                    i_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_vreg_count),
                                                WORKING_ADDRESS_A_REG,
                                                8,
                                                arr_spec_t::d));
                } else if (l_m_block[1] == 3) {
                    i_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_vreg_count),
                                                WORKING_ADDRESS_A_REG,
                                                8,
                                                arr_spec_t::d));

                    m_kernel.add_instr(neon_ld1_scalar_index(static_cast<simd_fp_t>(l_vreg_count),
                                                             WORKING_ADDRESS_A_REG,
                                                             2));
                    i_kernel.add_instr(base_add_imm(WORKING_ADDRESS_A_REG,
                                                    WORKING_ADDRESS_A_REG,
                                                    4,
                                                    0));
                }

                l_vreg_count += 1;
                l_vreg_count_a++;
            }

            // restore Working A
            m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_A_REG,
                                            WORKING_ADDRESS_A_REG,
                                            l_m_block[0] * 16 + l_m_block[1] * 4,
                                            0));

            int32_t l_b_vector_register = l_vreg_count;

            // compute with fmla
            for (size_t i = 0; i < i_used_reg_count; i++) {
                // load B value
                if (i % l_vreg_count_a == 0) {
                    m_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_b_vector_register),
                                                WORKING_ADDRESS_B_REG,
                                                0,
                                                arr_spec_t::s));
                    m_kernel.add_instr(base_add_shifted_register(WORKING_ADDRESS_B_REG,
                                                                 WORKING_ADDRESS_B_REG,
                                                                 LEADING_DIM_B_REG,
                                                                 0,
                                                                 0));
                }

                m_kernel.add_instr(neon_fmla_element(static_cast<simd_fp_t>(i),
                                                     static_cast<simd_fp_t>(i_used_reg_count + (i % l_vreg_count_a)),
                                                     static_cast<simd_fp_t>(l_b_vector_register),
                                                     element_spec_t::S4_0));
            }

            // restore Working B
            m_kernel.add_instr(base_mov_imm(HELP_REG_1,
                                            i_kernelsize.N,
                                            0));
            m_kernel.add_instr(base_mul_reg(HELP_REG_1,
                                            HELP_REG_1,
                                            LEADING_DIM_B_REG));
            m_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_B_REG,
                                                         WORKING_ADDRESS_B_REG,
                                                         HELP_REG_1,
                                                         0,
                                                         0));
        }
    };
}  // namespace TenGen::MiniJit::Generator

#endif  // TENGEN_MINI_JIT_GENERATOR_BRGEMM_H