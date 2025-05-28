#ifndef TENGEN_MINI_JIT_GENERATOR_UTIL_H
#define TENGEN_MINI_JIT_GENERATOR_UTIL_H

#include <cfloat>
#include <cstdint>
#include <string>
#include <vector>

#include "TenGen.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using namespace TenGen::MiniJit::Instructions::Encoding;
using Kernel = TenGen::MiniJit::Backend::Kernel;

namespace TenGen::MiniJit::Generator {
    class Util {
       private:
        static Kernel m_kernel;

        static uint32_t get_main_size(uint32_t n) {
            if (n == 0) return 0;
            n |= (n >> 1);
            n |= (n >> 2);
            n |= (n >> 4);
            n |= (n >> 8);
            n |= (n >> 16);
            return n - (n >> 1);
        }

        inline static const element_spec_t s4_values[] = {
            element_spec_t::S4_0,
            element_spec_t::S4_1,
            element_spec_t::S4_2,
            element_spec_t::S4_3};

       public:
        /**
         * @brief Get the four subarea sizes for a matrix C.
         *
         * @param i_m vertical dimensions of C.
         * @param i_n horizontal dimensions of C.
         * @param kernelsizes The size of each area.
         */
        static void get_area_sizes(int32_t m,
                                   int32_t n,
                                   std::vector<KernelSize> &work_areas) {
            KernelSize main_area = KernelSize{0, 0};
            KernelSize right_area = KernelSize{0, 0};
            KernelSize lower_area = KernelSize{0, 0};
            KernelSize remainder_area = KernelSize{0, 0};

            // split full matrix into working areas
            bool m_split = m < 16 || ((m >= 16) && (m % 4 == 0));
            bool n_split = n < 16 || ((n >= 16) && (n % 4 == 0));

            if (m_split && n_split) {
                // only main area
                main_area.M = m;
                main_area.N = n;

            } else if (m_split && !n_split) {
                // main and right
                main_area.M = m;
                main_area.N = get_main_size(n);
                right_area.M = m;
                right_area.N = n - main_area.N;
            } else if (!m_split && n_split) {
                // main and lower
                main_area.M = get_main_size(m);
                main_area.N = n;
                lower_area.M = m - main_area.M;
                lower_area.N = n;
            } else {
                // all areas
                main_area.M = get_main_size(m);
                main_area.N = get_main_size(n);
                right_area.M = main_area.M;
                right_area.N = n - main_area.N;
                lower_area.M = m - main_area.M;
                lower_area.N = main_area.N;
                remainder_area.M = lower_area.M;
                remainder_area.N = right_area.N;
            }
            work_areas.push_back(main_area);
            work_areas.push_back(right_area);
            work_areas.push_back(lower_area);
            work_areas.push_back(remainder_area);

            // std::cout << "MainArea Size      M: " << work_areas[0].M << ", N: " << work_areas[0].N << "\n"
            //<< "RightArea Size     M: " << work_areas[1].M << ", N: " << work_areas[1].N << "\n"
            //<< "LowerArea Size     M: " << work_areas[2].M << ", N: " << work_areas[2].N << "\n"
            //<< "RemainderArea Size M: " << work_areas[3].M << ", N: " << work_areas[3].N << "\n"
            //<< std::endl;
        }

        /**
         * @brief Get the four kernel sizes for the microkernels.
         *
         * @param i_m The number of rows in the matrix A.
         * @param i_n The number of columns in the matrix B.
         * @param kernelsizes The size of each kernel.
         */
        static void get_kernel_sizes(int32_t m,
                                     int32_t n,
                                     KernelSizes &kernelsizes,
                                     bool only_square = false) {
            int32_t max_reg_space = 32;

            std::cout << "M: " << m << ", N: " << n << std::endl;

            std::vector<KernelSize> work_areas;
            Util::get_area_sizes(m, n, work_areas);

            // define weights for scoring
            double w_rl = 1.2;
            double w_mn = 0.45;

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
                        if ((n_temp != m_temp) && only_square) continue;
                        // get used registers
                        int32_t A_regs = (m_temp - (m_temp % 4)) / 4 + ((m_temp % 4 == 0) ? 0 : 1);
                        int32_t B_regs = (n_temp - (n_temp % 4)) / 4 + ((n_temp % 4 == 0) ? 0 : 1);
                        int32_t C_size = m_temp * n_temp;
                        int32_t C_regs = (C_size - (C_size % 4) * n_temp) / 4 + ((C_size % 4 == 0) ? 0 : n_temp);
                        // int32_t C_regs = (C_size - (C_size % 4)) / 4 + (C_size % 4 == 0);
                        int32_t used_reg_space = A_regs + B_regs + C_regs;

                        if (max_reg_space >= used_reg_space && (area.M % m_temp == 0 && area.N % n_temp == 0)) {
                            // metrix for how much bigger m is compared to n
                            double n_greater_m_deficit = (double)n_temp / (double)m_temp;

                            // relative number of unused registers
                            double registers_left = (max_reg_space - used_reg_space) / (double)max_reg_space;

                            double score = w_rl * registers_left + w_mn * n_greater_m_deficit;
                            std::cout << m_temp << "x" << n_temp << " | " << score << ", rl: " << registers_left << ", nm: " << n_greater_m_deficit << std::endl;
                            if (score < min_score) {
                                std::cout << "taken" << std::endl;
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
            if (work_areas[1].M != 0) {
                kernelsizes.kernel2.M = kernelsizes_v[1].M;
                kernelsizes.kernel2.N = kernelsizes_v[1].N;
            } else {
                kernelsizes.kernel2.M = 0;
                kernelsizes.kernel2.N = 0;
                i = 1;
            }

            if (work_areas[2].M != 0) {
                kernelsizes.kernel3.M = kernelsizes_v[i].M;
                kernelsizes.kernel3.N = kernelsizes_v[i].N;
            } else {
                kernelsizes.kernel3.M = 0;
                kernelsizes.kernel3.N = 0;
            }

            if (work_areas[3].M != 0) {
                kernelsizes.kernel4.M = kernelsizes_v[3].M;
                kernelsizes.kernel4.N = kernelsizes_v[3].N;
            } else {
                kernelsizes.kernel4.M = 0;
                kernelsizes.kernel4.N = 0;
            }
        }
        /**
         * @brief Get the two kernel sizes for the microkernels and the loads/stores for the microkernel BRGEMM.
         */
        static void get_kernel_sizes_brgemm(int32_t m,
                                            int32_t n,
                                            KernelSize &kernelsize_big,
                                            KernelSize &kernelsize_small,
                                            int32_t &i_used_vector_reg_count_big,
                                            int32_t &i_used_vector_reg_count_small) {
            int32_t max_n_blocking = 30;
            int32_t m_blocks = 0;
            if (m > 12) {
                m_blocks = 4;
            } else {
                m_blocks = (m + 3) / 4;  // up_div
            }

            while (m_blocks * max_n_blocking + m_blocks + 1 > 32) {
                max_n_blocking--;
            }

            if (max_n_blocking > n) {
                max_n_blocking = n;
            }

            kernelsize_big.M = (m > 15) ? 16 : m;
            kernelsize_big.N = max_n_blocking;
            i_used_vector_reg_count_big = m_blocks * max_n_blocking;

            kernelsize_small.M = (m > 15) ? 16 : m;
            kernelsize_small.N = n % max_n_blocking;

            i_used_vector_reg_count_small = m_blocks * kernelsize_small.N;
        }

        /**
         * @brief Generate microkernels.
         * @param kernel The kernel sizes.
         * @param i_used_vector_reg_count The number of used vector registers.
         */
        static void gen_microkernel(KernelSize kernelsize,
                                    int32_t i_used_vector_reg_count) {
            // ----------------------------------------------------------------------------
            //
            // load A matrix (M elements)
            //

            // count how many vectors are in use, but skip already used ones
            int32_t reg_count = i_used_vector_reg_count;

            // total number of elements needed to load
            int count = kernelsize.M;
            int A_quads = count / 4;
            int rem = count % 4;

            // main quad (4s) loop
            for (; reg_count < A_quads + i_used_vector_reg_count; reg_count++) {
                // load four elements at once (4s)
                m_kernel.add_instr(
                    // TODO: THIS MUST BE REPLACED BY A WORKING LD1 FUNCTION
                    neon_ld1_no_offset(
                        static_cast<simd_fp_t>(reg_count),
                        WORKING_ADDRESS_A_REG,
                        vector_count_t::vc4));

                // advance the base pointer by 4 elements
                m_kernel.add_instr(
                    base_add_imm(
                        WORKING_ADDRESS_A_REG,
                        WORKING_ADDRESS_A_REG,
                        4,
                        /*no flags*/ 0));
            }

            // remainder
            for (int i = 0; i < rem; i++) {
                // load one element at a time (.s[N])
                m_kernel.add_instr(
                    neon_ld1_scalar_index(
                        static_cast<simd_fp_t>(reg_count),
                        WORKING_ADDRESS_A_REG,
                        i));

                // advance the base pointer by 1 elements
                m_kernel.add_instr(
                    base_add_imm(
                        WORKING_ADDRESS_A_REG,
                        WORKING_ADDRESS_A_REG,
                        1,
                        /*no flags*/ 0));
            }

            // ----------------------------------------------------------------------------

            // mark how many regs used for A
            // if there was a remainder, another register is used
            const int32_t regs_after_A = reg_count + 1;

            // ----------------------------------------------------------------------------
            //
            // load B matrix (N elements)
            //

            // load 4 elements in one register and than use antother regsiter
            for (int j = 0; j < kernelsize.N; j++) {
                m_kernel.add_instr(
                    neon_ld1_scalar_index(
                        static_cast<simd_fp_t>((j % 4 == 0) ? ++reg_count : reg_count),
                        WORKING_ADDRESS_B_REG,
                        j % 4));

                // advance the base pointer by K elements
                m_kernel.add_instr(
                    base_add_imm(
                        WORKING_ADDRESS_B_REG,
                        WORKING_ADDRESS_B_REG,
                        LEADING_DIM_B_REG,
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

            int has_remainder = (kernelsize.M % 4 == 0) ? 0 : 1;

            // for each col
            for (int j = 0; j < kernelsize.N; j++) {
                // after for values, use the next register for values in B
                if (j % 4 == 0) {
                    ++B_reg_count;
                }

                // for each row with each quad = (4s) + 1 if there is a remainder
                for (int i = 0; i < A_quads + has_remainder; i++) {
                    // mulitply four elements with one scalar (4s)
                    m_kernel.add_instr(
                        neon_fmla_element(static_cast<simd_fp_t>(C_reg_count++),
                                          static_cast<simd_fp_t>(i + i_used_vector_reg_count),
                                          static_cast<simd_fp_t>(B_reg_count + regs_after_A),
                                          s4_values[j % 4]));
                }
            }

            // DEBUG: write out / reset for debugging
            m_kernel.write("debug_gen_microkernel.bin");
            m_kernel.force_clear();
        }

        /**
         * @brief Load C block for the given kernel sizes.
         * @param kernel The kernel sizes.
         * @return used vector register count
         */
        static int32_t gen_matrix_load(Kernel &i_kernel, KernelSize kernelsize, gpr_t pointer_register, uint32_t leading_dimension) {
            // count how many vectors are in use
            int32_t reg_count = 0;

            // total number of elements needed to load
            int count = kernelsize.M;
            int quads = count / 4;
            int rem = count % 4;

            uint32_t ld_bytes = leading_dimension * 4;

            // for each col
            for (int j = 0; j < kernelsize.N; j++) {
                // for each row with each quad = (4s)
                for (int i = 0; i < quads; i++) {
                    // load four elements at once (4s)
                    i_kernel.add_instr(
                        neon_ld1_no_offset(
                            static_cast<simd_fp_t>(reg_count++),
                            pointer_register,
                            vector_count_t::vc4));

                    // advance the base pointer by 4 elements
                    i_kernel.add_instr(
                        base_add_imm(
                            pointer_register,
                            pointer_register,
                            16,
                            /*no flags*/ 0));
                }

                // remainder
                for (int i = 0; i < rem; i++) {
                    // load one element at a time (.s[N])
                    i_kernel.add_instr(
                        neon_ld1_scalar_index(
                            static_cast<simd_fp_t>(reg_count),
                            pointer_register,
                            i));

                    // advance the base pointer by 1 elements
                    i_kernel.add_instr(
                        base_add_imm(
                            pointer_register,
                            pointer_register,
                            4,
                            /*no flags*/ 0));
                }

                i_kernel.add_instr(
                    base_sub_imm(
                        pointer_register,
                        pointer_register,
                        kernelsize.M * 4,
                        /*no flags*/ 0));

                i_kernel.add_instr(
                    base_add_imm(
                        pointer_register,
                        pointer_register,
                        ld_bytes,
                        /*no flags*/ 0));

                (0 < rem) ? ++reg_count : reg_count;
            }

            i_kernel.add_instr(
                base_sub_imm(
                    pointer_register,
                    pointer_register,
                    ld_bytes * (kernelsize.N),
                    /*no flags*/ 0));

            // DEBUG: write out / reset for debugging
            /*m_kernel.write("debug_load_C.bin");
            m_kernel.force_clear();*/

            // if there was a remainder, another register is used
            return reg_count;
        }

        /**
         * @brief Store C block for the given kernel sizes.
         * @param kernel The kernel sizes.
         */
        static void gen_matrix_store(Kernel &i_kernel, KernelSize kernelsize, gpr_t pointer_register, uint32_t leading_dimension) {
            // count how many vectors are in use
            int32_t reg_count = 0;

            // total number of elements needed to load
            int count = kernelsize.M;
            int quads = count / 4;
            int rem = count % 4;

            uint32_t ld_bytes = leading_dimension * 4;

            // for each col
            for (int j = 0; j < kernelsize.N; j++) {
                // for each row with each quad = (4s)
                for (int i = 0; i < quads; i++) {
                    // load four elements at once (4s)
                    i_kernel.add_instr(
                        neon_st1_no_offset(
                            static_cast<simd_fp_t>(reg_count++),
                            pointer_register,
                            vector_count_t::vc4));

                    // advance the base pointer by 4 elements
                    i_kernel.add_instr(
                        base_add_imm(
                            pointer_register,
                            pointer_register,
                            16,
                            /*no flags*/ 0));
                }

                // remainder
                for (int i = 0; i < rem; i++) {
                    // load one element at a time (.s[N])
                    i_kernel.add_instr(
                        neon_st1_scalar_index(
                            static_cast<simd_fp_t>(reg_count),
                            pointer_register,
                            i));

                    // advance the base pointer by 1 elements
                    i_kernel.add_instr(
                        base_add_imm(
                            pointer_register,
                            pointer_register,
                            4,
                            /*no flags*/ 0));
                }

                i_kernel.add_instr(
                    base_sub_imm(
                        pointer_register,
                        pointer_register,
                        kernelsize.M * 4,
                        /*no flags*/ 0));

                i_kernel.add_instr(
                    base_add_imm(
                        pointer_register,
                        pointer_register,
                        ld_bytes,
                        /*no flags*/ 0));

                (0 < rem) ? ++reg_count : reg_count;
            }

            i_kernel.add_instr(
                base_sub_imm(
                    pointer_register,
                    pointer_register,
                    ld_bytes * (kernelsize.N),
                    /*no flags*/ 0));

            // DEBUG: write out / reset for debugging
            /*m_kernel.write("debug_store_C.bin");
            m_kernel.force_clear();*/
        }

        /**
         * @brief Load a block of B with the given kernel size to vector registers
         */
        static void generator_load_reg_block(Kernel &i_kernel,
                                             KernelSize &i_kernelsize,
                                             gpr_t i_register) {
            int32_t l_n_count = 0;
            int32_t l_reg_count = 0;

            // prepare C restore register Help 1
            i_kernel.add_instr(base_mov_imm(HELP_REG_1, i_kernelsize.N, 0));
            i_kernel.add_instr(base_mul_reg(HELP_REG_1, HELP_REG_1, LEADING_DIM_C_REG));

            for (; l_n_count < i_kernelsize.N; l_n_count++) {
                int32_t l_m_count = 0;
                while (l_m_count < i_kernelsize.M) {
                    if ((i_kernelsize.M - l_m_count) > 15) {
                        i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc4));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        64,
                                                        0));
                        l_reg_count += 4;
                        l_m_count += 16;
                    } else if ((i_kernelsize.M - l_m_count) > 11) {
                        i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc3));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        48,
                                                        0));
                        l_reg_count += 3;
                        l_m_count += 12;
                    } else if ((i_kernelsize.M - l_m_count) > 7) {
                        i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc2));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        32,
                                                        0));
                        l_reg_count += 2;
                        l_m_count += 8;
                    } else if ((i_kernelsize.M - l_m_count) > 3) {
                        i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc1));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        16,
                                                        0));
                        l_reg_count += 1;
                        l_m_count += 4;
                    } else {
                        if ((i_kernelsize.M - l_m_count) > 1) {
                            i_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_reg_count),
                                                        i_register,
                                                        0,
                                                        arr_spec_t::d));
                            i_kernel.add_instr(base_add_imm(i_register,
                                                            i_register,
                                                            8,
                                                            0));
                            l_m_count += 2;
                        }
                        if (((i_kernelsize.M - l_m_count) % 2) == 1) {
                            i_kernel.add_instr(neon_ld1_scalar_index(static_cast<simd_fp_t>(l_reg_count),
                                                                     i_register,
                                                                     (i_kernelsize.M % 4 == 3) ? 2 : 0));
                            i_kernel.add_instr(base_add_imm(i_register,
                                                            i_register,
                                                            4,
                                                            0));
                            l_m_count++;
                        }
                        l_reg_count += 1;
                    }
                }
                // Add Leading dimension
                i_kernel.add_instr(base_sub_imm(i_register,
                                                i_register,
                                                i_kernelsize.M * 4,
                                                0));
                i_kernel.add_instr(base_add_shifted_register(i_register,
                                                             i_register,
                                                             LEADING_DIM_C_REG,
                                                             0,
                                                             0));
            }
            // restore Working C reg for now with
            i_kernel.add_instr(base_sub_shifted_register(i_register,
                                                         i_register,
                                                         HELP_REG_1,
                                                         0,
                                                         0));
        }

        /**
         * @brief Store a block of B with the given kernel size to vector registers
         */
        static void generator_store_reg_block(Kernel &i_kernel,
                                              KernelSize &i_kernelsize,
                                              gpr_t i_register) {
            int32_t l_n_count = 0;
            int32_t l_reg_count = 0;

            // prepare C restore register Help 1
            i_kernel.add_instr(base_mov_imm(HELP_REG_1, i_kernelsize.N, 0));
            i_kernel.add_instr(base_mul_reg(HELP_REG_1, HELP_REG_1, LEADING_DIM_C_REG));

            for (; l_n_count < i_kernelsize.N; l_n_count++) {
                int32_t l_m_count = 0;
                while (l_m_count < i_kernelsize.M) {
                    if ((i_kernelsize.M - l_m_count) > 15) {
                        i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc4));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        64,
                                                        0));
                        l_reg_count += 4;
                        l_m_count += 16;
                    } else if ((i_kernelsize.M - l_m_count) > 11) {
                        i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc3));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        48,
                                                        0));
                        l_reg_count += 3;
                        l_m_count += 12;
                    } else if ((i_kernelsize.M - l_m_count) > 7) {
                        i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc2));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        32,
                                                        0));
                        l_reg_count += 2;
                        l_m_count += 8;
                    } else if ((i_kernelsize.M - l_m_count) > 3) {
                        i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                              i_register,
                                                              vector_count_t::vc1));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        16,
                                                        0));
                        l_reg_count += 1;
                        l_m_count += 4;
                    } else {
                        if ((i_kernelsize.M - l_m_count) > 1) {
                            i_kernel.add_instr(neon_str(static_cast<simd_fp_t>(l_reg_count),
                                                        i_register,
                                                        0,
                                                        arr_spec_t::d));
                            i_kernel.add_instr(base_add_imm(i_register,
                                                            i_register,
                                                            8,
                                                            0));
                            l_m_count += 2;
                        }
                        if (((i_kernelsize.M - l_m_count) % 2) == 1) {
                            i_kernel.add_instr(neon_st1_scalar_index(static_cast<simd_fp_t>(l_reg_count),
                                                                     i_register,
                                                                     (i_kernelsize.M % 4 == 3) ? 2 : 0));
                            i_kernel.add_instr(base_add_imm(i_register,
                                                            i_register,
                                                            4,
                                                            0));
                            l_m_count++;
                        }

                        l_reg_count += 1;
                    }
                }
                // Add Leading dimension
                i_kernel.add_instr(base_sub_imm(i_register,
                                                i_register,
                                                i_kernelsize.M * 4,
                                                0));
                i_kernel.add_instr(base_add_shifted_register(i_register,
                                                             i_register,
                                                             LEADING_DIM_C_REG,
                                                             0,
                                                             0));
            }
            // restore Working C reg for now with
            i_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_C_REG,
                                                         WORKING_ADDRESS_C_REG,
                                                         HELP_REG_1,
                                                         0,
                                                         0));
        }
    };
}  // namespace TenGen::MiniJit::Generator
#endif  // TENGEN_MINI_JIT_GENERATOR_UTIL_H
