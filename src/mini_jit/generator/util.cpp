#include "util.h"

#include <float.h>
#include <math.h>

#include <iostream>
#include <vector>

#include "../instructions/instructions.h"

using mini_jit::instructions::InstGen;

uint32_t get_main_size(uint32_t n) {
    if (n == 0) return 0;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return n - (n >> 1);
}

static const InstGen::element_spec_t s4_values[] = {
    InstGen::S4_0,
    InstGen::S4_1,
    InstGen::S4_2,
    InstGen::S4_3};
}

namespace mini_jit::generator {

    mini_jit::backend::Kernel Util::m_kernel;

    void mini_jit::generator::Util::get_kernel_sizes(int32_t m,
                                                     int32_t n,
                                                     mini_jit::generator::Util::KernelSizes &kernelsizes) {
        int32_t max_reg_space = 32;

        std::cout << "M: " << m << ", N: " << n << std::endl;

        KernelSize main_area = KernelSize{0, 0};
        KernelSize right_area = KernelSize{0, 0};
        KernelSize lower_area = KernelSize{0, 0};
        KernelSize remainder_area = KernelSize{0, 0};
        std::vector<KernelSize> work_areas;

        // split full matrix into working areas
        bool m_split = m < 16 || ((m >= 16) && (m % 4 == 0));
        bool n_split = n < 16 || ((n >= 16) && (n % 4 == 0));

        if (m_split && n_split) {
            main_area.M = m;
            main_area.N = n;
            work_areas.push_back(main_area);
        } else if (m_split && !n_split) {
            main_area.M = m;
            main_area.N = get_main_size(n);
            right_area.M = m;
            right_area.N = n - main_area.N;
            work_areas.push_back(main_area);
            work_areas.push_back(right_area);
        } else if (!m_split && n_split) {
            main_area.M = get_main_size(m);
            main_area.N = n;
            lower_area.M = m - main_area.M;
            lower_area.N = n;
            work_areas.push_back(main_area);
            work_areas.push_back(lower_area);
        } else {
            main_area.M = get_main_size(m);
            main_area.N = get_main_size(n);
            right_area.M = main_area.M;
            right_area.N = n - main_area.N;
            lower_area.M = m - main_area.M;
            lower_area.N = main_area.N;
            remainder_area.M = lower_area.M;
            remainder_area.N = right_area.N;
            work_areas.push_back(main_area);
            work_areas.push_back(right_area);
            work_areas.push_back(lower_area);
            work_areas.push_back(remainder_area);
        }

        std::cout << "MainArea      M: " << main_area.M << ", N: " << main_area.N << "\n"
                  << "RightArea     M: " << right_area.M << ", N: " << right_area.N << "\n"
                  << "LowerArea     M: " << lower_area.M << ", N: " << lower_area.N << "\n"
                  << "RemainderArea M: " << remainder_area.M << ", N: " << remainder_area.N << "\n"
                  << std::endl;

        // define weights for scoring
        double w_sd = 0.1;
        double w_rl = 0.3;
        double w_mn = 0.3;

        // find a kernel for each working areas
        std::vector<KernelSize> kernelsizes_v;
        for (KernelSize area : work_areas) {
            double min_score = DBL_MAX;
            int32_t best_m = 0;
            int32_t best_n = 0;

            // do not choose kernels, that are bigger than area
            int32_t m_upper = (area.M < 16) ? area.M : 16;
            int32_t n_upper = (area.N < 16) ? area.N : 16;

            for (int32_t m_temp = 1; m_temp <= m_upper; m_temp++) {
                // iterate only to m_temp as more as a biggeer n means more B load instructions
                for (int32_t n_temp = 1; n_temp <= n_upper; n_temp++) {
                    // get used registers
                    int32_t A_regs = (m_temp - (m_temp % 4)) / 4 + ((m_temp % 4 == 0) ? 0 : 1);
                    int32_t B_regs = (n_temp - (n_temp % 4)) / 4 + ((n_temp % 4 == 0) ? 0 : 1);
                    // int32_t B_regs = n_temp;
                    int32_t C_size = m_temp * n_temp;
                    // int32_t C_regs = (C_size - (C_size % 4)) / 4 + ((C_size % 4 == 0) ? 0 : 1);
                    int32_t C_regs = (C_size - (C_size % 4)) / 4 + (C_size % 4 == 0);
                    int32_t used_reg_space = A_regs + B_regs + C_regs;

                    if (max_reg_space >= used_reg_space && (area.M % m_temp == 0 && area.N % n_temp == 0)) {
                        // metric for how square the rectangle spanned by n_temp and m_temp is
                        double squareness_deficit = fabs(((double)n_temp / (double)m_temp) - 1);

                        // metrix for how much bigger m is compared to n
                        double n_greater_m_deficit = (double)n_temp / (double)m_temp;

                        // relative number of unused registers
                        double registers_left = (max_reg_space - used_reg_space) / (double)max_reg_space;

                        double score = w_sd * squareness_deficit + w_rl * registers_left + w_mn * n_greater_m_deficit;

                        if (score < min_score) {
                            min_score = score;
                            best_m = m_temp;
                            best_n = n_temp;
                        }
                    }
                }
            }
            kernelsizes_v.push_back(KernelSize{best_m, best_n});
        }

        // fill return objekt
        kernelsizes.kernel1.M = kernelsizes_v[0].M;
        kernelsizes.kernel1.N = kernelsizes_v[0].N;

        int i = 2;
        if (right_area.M != 0) {
            kernelsizes.kernel2.M = kernelsizes_v[1].M;
            kernelsizes.kernel2.N = kernelsizes_v[1].N;
        } else {
            kernelsizes.kernel2.M = 0;
            kernelsizes.kernel2.N = 0;
            i = 1;
        }

        if (lower_area.M != 0) {
            kernelsizes.kernel3.M = kernelsizes_v[i].M;
            kernelsizes.kernel3.N = kernelsizes_v[i].N;
        } else {
            kernelsizes.kernel3.M = 0;
            kernelsizes.kernel3.N = 0;
        }

        if (remainder_area.M != 0) {
            kernelsizes.kernel4.M = kernelsizes_v[3].M;
            kernelsizes.kernel4.N = kernelsizes_v[3].N;
        } else {
            kernelsizes.kernel4.M = 0;
            kernelsizes.kernel4.N = 0;
        }

        std::cout << "MainArea Kernel      M: " << kernelsizes.kernel1.M << ", N: " << kernelsizes.kernel1.N << "\n"
                  << "RightArea Kernel     M: " << kernelsizes.kernel2.M << ", N: " << kernelsizes.kernel2.N << "\n"
                  << "LowerArea Kernel     M: " << kernelsizes.kernel3.M << ", N: " << kernelsizes.kernel3.N << "\n"
                  << "RemainderArea Kernel M: " << kernelsizes.kernel4.M << ", N: " << kernelsizes.kernel4.N << "\n"
                  << std::endl;
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
