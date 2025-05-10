#include "util.h"

#include <iostream>

#include "../instructions/instructions.h"

using mini_jit::instructions::InstGen;

namespace mini_jit::generator {

    mini_jit::backend::Kernel Util::m_kernel;

    void Util::get_kernel_sizes(int32_t i_m, int32_t i_n, Util::KernelSizes kernelsizes) {
    }

    void Util::gen_microkernel(Util::KernelSize kernelsize, int32_t i_used_vector_reg_count) {
        // start allocating vector registers from here
        int32_t reg_index = i_used_vector_reg_count;

        //
        // load A matrix (M elements)
        //
        for (int remaining = kernelsize.M; remaining > 0; ++reg_index) {
            // grab up to 4 lanes at a time
            const auto count = std::min(remaining, 4);
            remaining -= count;

            m_kernel.add_instr(
                InstGen::neon_ld1_no_offset(
                    static_cast<InstGen::simd_fp_t>(reg_index),
                    Util::WORKING_ADDRESS_A_REG,
                    static_cast<InstGen::vector_count_t>(count)));
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_A_REG,
                    Util::WORKING_ADDRESS_A_REG,
                    count,
                    /*no flags*/ 0));
        }
        // mark how many regs used for A
        const int32_t regs_after_A = reg_index;

        //
        // load B matrix (N elements)
        //
        for (int remaining = kernelsize.N; remaining > 0; ++reg_index) {
            const auto count = std::min(remaining, 4);
            remaining -= count;

            m_kernel.add_instr(
                InstGen::neon_ld1_no_offset(
                    static_cast<InstGen::simd_fp_t>(reg_index),
                    Util::WORKING_ADDRESS_B_REG,
                    static_cast<InstGen::vector_count_t>(count)));
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_B_REG,
                    Util::WORKING_ADDRESS_B_REG,
                    count,
                    /*no flags*/ 0));
        }

        //
        // FMA
        //

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_gen_microkernel.bin");
        m_kernel.force_clear();
    }

    // I assume that get_kernel_size only return valid kernels, so there must be enough registers
    // to fit each value in and that the working registers are aligned already.
    int32_t Util::gen_c_load(Util::KernelSize kernelsize) {
        // total number of elements we need to load
        int remaining = kernelsize.M * kernelsize.N;

        // count how many vector instructions we'll emit
        int32_t load_ops = 0;

        // loop until we've consumed all elements
        for (; remaining > 0; ++load_ops) {
            // grab up to 4 elements at a time
            const auto count = std::min(remaining, 4);
            remaining -= count;

            // issue the NEON load instruction
            m_kernel.add_instr(
                InstGen::neon_ld1_no_offset(
                    static_cast<InstGen::simd_fp_t>(load_ops),
                    Util::WORKING_ADDRESS_C_REG,
                    static_cast<InstGen::vector_count_t>(count)));

            // advance the base pointer by the same count
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_C_REG,
                    Util::WORKING_ADDRESS_C_REG,
                    count,
                    /*no flags*/ 0));
        }

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_load_C.bin");
        m_kernel.force_clear();

        return load_ops;
    }

    void Util::gen_c_store(Util::KernelSize kernelsize) {
        // total number of elements to store
        int remaining = kernelsize.M * kernelsize.N;

        // count how many store instructions we'll emit
        int32_t store_ops = 0;

        // loop until all elements are stored
        for (; remaining > 0; ++store_ops) {
            // take up to 4 elements per NEON store
            const auto count = std::min(remaining, 4);
            remaining -= count;

            // issue the NEON store instruction
            m_kernel.add_instr(
                InstGen::neon_st1_no_offset(
                    static_cast<InstGen::simd_fp_t>(store_ops),
                    Util::WORKING_ADDRESS_C_REG,
                    static_cast<InstGen::vector_count_t>(count)));

            // advance the base pointer by the same count
            m_kernel.add_instr(
                InstGen::base_add_imm(
                    Util::WORKING_ADDRESS_C_REG,
                    Util::WORKING_ADDRESS_C_REG,
                    count,
                    /*no flags*/ 0));
        }

        // DEBUG: write out / reset for debugging
        m_kernel.write("debug_store_C.bin");
        m_kernel.force_clear();
    }
}  // namespace mini_jit::generator
