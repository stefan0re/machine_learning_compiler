#include "util.h"

#include <iostream>

#include "../instructions/instructions.h"

using mini_jit::instructions::InstGen;

static const InstGen::element_spec_t s4_values[] = {
    InstGen::S4_0,
    InstGen::S4_1,
    InstGen::S4_2,
    InstGen::S4_3};

namespace mini_jit::generator {

    mini_jit::backend::Kernel Util::m_kernel;

    void Util::get_kernel_sizes(int32_t i_m, int32_t i_n, Util::KernelSizes kernelsizes) {
    }

    void Util::gen_microkernel(Util::KernelSize kernelsize, int K, int32_t i_used_vector_reg_count) {
        // ----------------------------------------------------------------------------
        //
        // load A matrix (M elements)
        //

        // count how many vectors are in use, but scipt already used ones
        int32_t reg_count = i_used_vector_reg_count;

        // total number of elements needed to load
        int count = kernelsize.M;
        int A_quads = count / 4;
        int rem = count % 4;

        // main quad (4s) loop
        for (; reg_count < A_quads + i_used_vector_reg_count; reg_count++) {
            // load four elements at once (4s)
            m_kernel.add_instr(
                InstGen::neon_ld1_no_offset(
                    static_cast<InstGen::simd_fp_t>(reg_count),
                    Util::WORKING_ADDRESS_A_REG,
                    InstGen::vector_count_t::vc4));

            // advance the base pointer by 4 elements
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_A_REG,
                    Util::WORKING_ADDRESS_A_REG,
                    4,
                    /*no flags*/ 0));
        }

        // remainder
        for (int i = 0; i < rem; i++) {
            // load one element at a time (.s[N])
            m_kernel.add_instr(
                InstGen::neon_ld1_scalar_index(
                    static_cast<InstGen::simd_fp_t>(reg_count),
                    Util::WORKING_ADDRESS_A_REG,
                    i));

            // advance the base pointer by 1 elements
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_A_REG,
                    Util::WORKING_ADDRESS_A_REG,
                    1,
                    /*no flags*/ 0));
        }

        // ----------------------------------------------------------------------------

        // mark how many regs used for A
        // if there was a remainder, another register is used
        const int32_t regs_after_A = (0 < rem) ? ++reg_count : reg_count;

        // ----------------------------------------------------------------------------
        //
        // load B matrix (N elements)
        //

        // total number of elements needed to load
        count = kernelsize.N;
        int B_quads = count / 4;
        rem = count % 4;

        // main quad (4s) loop
        for (; reg_count < B_quads + regs_after_A; reg_count++) {
            // load four elements at once (4s)
            m_kernel.add_instr(
                InstGen::neon_ld1_no_offset(
                    static_cast<InstGen::simd_fp_t>(reg_count),
                    Util::WORKING_ADDRESS_B_REG,
                    InstGen::vector_count_t::vc4));

            // advance the base pointer by K elements
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_B_REG,
                    Util::WORKING_ADDRESS_B_REG,
                    K,
                    /*no flags*/ 0));
        }

        // remainder
        for (int i = 0; i < rem; i++) {
            // load one element at a time (.s[N])
            m_kernel.add_instr(
                InstGen::neon_ld1_scalar_index(
                    static_cast<InstGen::simd_fp_t>(reg_count),
                    Util::WORKING_ADDRESS_B_REG,
                    i));

            // advance the base pointer by K elements
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_B_REG,
                    Util::WORKING_ADDRESS_B_REG,
                    K,
                    /*no flags*/ 0));
        }

        // ----------------------------------------------------------------------------

        // mark how many regs used for A
        // if there was a remainder, another register is used
        const int32_t regs_after_A_and_B = (0 < rem) ? ++reg_count : reg_count;

        // ----------------------------------------------------------------------------
        //
        // FMA
        //

        // count how many vectors are in use
        int C_reg_count = 0;
        int B_reg_count = -1;

        // total number of elements needed to load
        count = kernelsize.M;
        rem = count % 4;

        // for each col
        for (int j = 0; j < kernelsize.N; j++) {
            // after for values, use the next register for values in B
            (j % 4 == 0) ? ++B_reg_count : 0;

            // for each row with each quad = (4s) + 1 for remainder
            for (int i = 0; i < A_quads + 1; i++) {
                // mulitply four elements with one scalar (4s)
                m_kernel.add_instr(
                    InstGen::neon_fmla_element(static_cast<InstGen::simd_fp_t>(C_reg_count++),
                                               static_cast<InstGen::simd_fp_t>(i + i_used_vector_reg_count),
                                               static_cast<InstGen::simd_fp_t>(B_reg_count + regs_after_A),
                                               s4_values[j % 4]));
            }
        }

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_gen_microkernel.bin");
        m_kernel.force_clear();
    }

    // I assume that get_kernel_size only return valid kernels, so there must be enough registers
    // to fit each value in and that the working registers are aligned already.
    int32_t Util::gen_c_load(Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;
        int rem = count % 4;

        // for each col
        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                // load four elements at once (4s)
                m_kernel.add_instr(
                    InstGen::neon_ld1_no_offset(
                        static_cast<InstGen::simd_fp_t>(reg_count++),
                        Util::WORKING_ADDRESS_C_REG,
                        InstGen::vector_count_t::vc4));

                // advance the base pointer by 4 elements
                m_kernel.add_instr(
                    InstGen::base_add_imm(
                        Util::WORKING_ADDRESS_C_REG,
                        Util::WORKING_ADDRESS_C_REG,
                        4,
                        /*no flags*/ 0));
            }

            // remainder
            for (int i = 0; i < rem; i++) {
                // load one element at a time (.s[N])
                m_kernel.add_instr(
                    InstGen::neon_ld1_scalar_index(
                        static_cast<InstGen::simd_fp_t>(reg_count),
                        Util::WORKING_ADDRESS_C_REG,
                        i));

                // advance the base pointer by 1 elements
                m_kernel.add_instr(
                    InstGen::base_add_imm(
                        Util::WORKING_ADDRESS_C_REG,
                        Util::WORKING_ADDRESS_C_REG,
                        1,
                        /*no flags*/ 0));
            }

            (0 < rem) ? ++reg_count : reg_count;
        }

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_load_C.bin");
        m_kernel.force_clear();

        // if there was a remainder, another register is used
        return reg_count;
    }

    void Util::gen_c_store(Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;
        int rem = count % 4;

        // for each col
        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                // load four elements at once (4s)
                m_kernel.add_instr(
                    InstGen::neon_st1_no_offset(
                        static_cast<InstGen::simd_fp_t>(reg_count++),
                        Util::WORKING_ADDRESS_C_REG,
                        InstGen::vector_count_t::vc4));

                // advance the base pointer by 4 elements
                m_kernel.add_instr(
                    InstGen::base_add_imm(
                        Util::WORKING_ADDRESS_C_REG,
                        Util::WORKING_ADDRESS_C_REG,
                        4,
                        /*no flags*/ 0));
            }

            // remainder
            for (int i = 0; i < rem; i++) {
                // load one element at a time (.s[N])
                m_kernel.add_instr(
                    InstGen::neon_st1_scalar_index(
                        static_cast<InstGen::simd_fp_t>(reg_count),
                        Util::WORKING_ADDRESS_C_REG,
                        i));

                // advance the base pointer by 1 elements
                m_kernel.add_instr(
                    InstGen::base_add_imm(
                        Util::WORKING_ADDRESS_C_REG,
                        Util::WORKING_ADDRESS_C_REG,
                        1,
                        /*no flags*/ 0));
            }

            (0 < rem) ? ++reg_count : reg_count;
        }

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_store_C.bin");
        m_kernel.force_clear();
    }
}  // namespace mini_jit::generator
