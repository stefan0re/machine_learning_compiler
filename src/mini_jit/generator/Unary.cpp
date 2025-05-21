#include "Unary.h"

#include <float.h>
#include <math.h>

#include <iostream>

#include "../instructions/instructions.h"
#include "Util.h"

namespace inst = mini_jit::instructions;

namespace mini_jit::generator {
    typedef struct {
        int32_t m;
        int32_t n;
        int32_t m_iters;
        int32_t n_iters;
        uint32_t offset;
        Util::KernelSize kernelsize;
    } AreaDefinition;

    mini_jit::backend::Kernel Unary::m_kernel;

    void Unary::gen_unary_zero(mini_jit::generator::Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;

        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
            }
        }
    }

    Unary::error_t Unary::get_kernel_sizes(uint32_t m,
                                           uint32_t n,
                                           Util::KernelSizes& kernelsizes) {
        int32_t max_reg_space = 32;

        std::vector<Util::KernelSize> work_areas;
        Util::get_area_sizes(m, n, work_areas);

        // define weights for scoring
        double w_sd = 0.1;
        double w_rl = 0.3;
        double w_mn = 0.3;

        // find a kernel for each working areas
        std::vector<Util::KernelSize> kernelsizes_v;
        for (Util::KernelSize area : work_areas) {
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
                    int32_t mxn = m_temp * n_temp;
                    int32_t A_regs = (mxn - (mxn % 4)) / 4 + ((mxn % 4 == 0) ? 0 : 1);
                    int32_t B_regs = (mxn - (mxn % 4)) / 4 + ((mxn % 4 == 0) ? 0 : 1);

                    int32_t used_reg_space = A_regs + B_regs;

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
            kernelsizes_v.push_back(Util::KernelSize{best_m, best_n});
        }

        // fill return objekt
        kernelsizes.kernel1.M = kernelsizes_v[0].M;
        kernelsizes.kernel1.N = kernelsizes_v[0].N;

        kernelsizes.kernel2.M = kernelsizes_v[1].M;
        kernelsizes.kernel2.N = kernelsizes_v[1].N;

        kernelsizes.kernel3.M = kernelsizes_v[2].M;
        kernelsizes.kernel3.N = kernelsizes_v[2].N;

        kernelsizes.kernel4.M = kernelsizes_v[3].M;
        kernelsizes.kernel4.N = kernelsizes_v[3].N;

        std::cout << "MainArea Kernel      M: " << kernelsizes.kernel1.M << ", N: " << kernelsizes.kernel1.N << "\n"
                  << "RightArea Kernel     M: " << kernelsizes.kernel2.M << ", N: " << kernelsizes.kernel2.N << "\n"
                  << "LowerArea Kernel     M: " << kernelsizes.kernel3.M << ", N: " << kernelsizes.kernel3.N << "\n"
                  << "RemainderArea Kernel M: " << kernelsizes.kernel4.M << ", N: " << kernelsizes.kernel4.N << "\n"
                  << std::endl;

        return Unary::error_t::success;
    }

    Unary::error_t Unary::generate(uint32_t m,
                                   uint32_t n,
                                   uint32_t trans_b,
                                   Unary::dtype_t dtype,
                                   Unary::ptype_t ptype) {
        Util::KernelSizes kernels;
        Unary::get_kernel_sizes(m, n, kernels);

        // get area definations
        std::vector<AreaDefinition> areas;

        // get iterations per kernel
        int32_t remainder_m_size = (int32_t)m % (int32_t)kernels.kernel1.M;
        int32_t remainder_n_size = (int32_t)n % (int32_t)kernels.kernel1.N;
        int32_t main_m_size = (int32_t)m - remainder_m_size;
        int32_t main_n_size = (int32_t)n - remainder_n_size;

        int32_t main_m_iters = (int32_t)(main_m_size / (double)kernels.kernel1.M);
        int32_t main_n_iters = (int32_t)(main_n_size / (double)kernels.kernel1.N);
        areas.push_back({main_m_size, main_n_size, main_m_iters, main_n_iters, 0, kernels.kernel1});
        std::cout << "Main Def:\nsizes: m=" << main_m_size << ", n=" << main_n_size << "\niters: m=" << main_m_iters << ", n=" << main_n_iters << "\noffset=" << 0 << std::endl;

        int32_t right_m_iters = (int32_t)(remainder_n_size != 0) * (int32_t)(main_m_size / (double)kernels.kernel2.M);
        int32_t lower_n_iters = (int32_t)(remainder_m_size != 0) * (int32_t)(main_n_size / (double)kernels.kernel3.N);

        if (right_m_iters != 0) {
            uint32_t offset = main_n_size * m * 4;
            std::cout << "Right Def:\nsizes: m=" << main_m_size << ", n=" << remainder_n_size << "\niters: m=" << right_m_iters << ", n=" << 1 << "\noffset=" << offset << std::endl;
            areas.push_back({main_n_size, remainder_n_size, right_m_iters, 1, offset, kernels.kernel2});
        }

        if (lower_n_iters != 0) {
            uint32_t offset = main_m_size * 4;
            std::cout << "Lower Def:\nsizes: m=" << remainder_m_size << ", n=" << main_n_size << "\niters: m=" << 1 << ", n=" << lower_n_iters << "\noffset=" << offset << std::endl;
            areas.push_back({main_n_size, remainder_n_size, 1, lower_n_iters, offset, kernels.kernel3});
        }

        if (lower_n_iters != 0 && right_m_iters != 0) {
            uint32_t offset = main_n_size * m * 4 + main_m_size * 4;
            std::cout << "Lower Def:\nsizes: m=" << remainder_m_size << ", n=" << remainder_n_size << "\niters: m=" << 1 << ", n=" << 1 << "\noffset=" << offset << std::endl;
            areas.push_back({remainder_m_size, remainder_n_size, 1, 1, offset, kernels.kernel4});
        }

        // procedure call standard (store to stack)
        m_kernel.add_instr(0x6DBF27E8);
        m_kernel.add_instr(0x6DBF2FEA);
        m_kernel.add_instr(0x6DBF37EC);
        m_kernel.add_instr(0x6DBF3FEE);

        for (AreaDefinition area : areas) {
            // Store pointers of A and B to x7, x8
            m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x7,
                                                                inst::InstGen::x0));
            m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x8,
                                                                inst::InstGen::x1));

            // add offset for working area
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7,
                                                           inst::InstGen::x7,
                                                           (int32_t)area.offset,
                                                           0));

            // shift leading dimensions to 4 bytes
            m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x2, inst::InstGen::x2, 2));
            m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x3, inst::InstGen::x3, 2));

            // N Counter
            m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x9, area.n_iters, 0));

            // N loop
            size_t n_loop_count = m_kernel.get_size();

            // decrease N counter
            m_kernel.add_instr(
                inst::InstGen::base_sub_imm(
                    inst::InstGen::x9,
                    inst::InstGen::x9,
                    1,
                    /*no flags*/ 0));

            // M Counter
            m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x10, area.m_iters, 0));

            // M loop
            size_t m_loop_count = m_kernel.get_size();

            // decrease M counter
            m_kernel.add_instr(
                inst::InstGen::base_sub_imm(
                    inst::InstGen::x10,
                    inst::InstGen::x10,
                    1,
                    /*no flags*/ 0));

            // load A kernel
            int32_t regs_used_l = Util::gen_matrix_load(m_kernel, area.kernelsize, inst::InstGen::x7, m);

            if (ptype == Unary::ptype_t::zero) {
                Unary::gen_unary_zero(area.kernelsize);
            }

            // store in B
            Util::gen_matrix_store(m_kernel, area.kernelsize, inst::InstGen::x8, m);

            // move pointers in M dimension
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, area.kernelsize.M * 4, 0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, area.kernelsize.M * 4, 0));

            // jump M loop
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x10, (m_loop_count - m_kernel.get_size()) / 4));

            // move to root
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x7, inst::InstGen::x7, area.m * 4, 0));
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x8, inst::InstGen::x8, area.m * 4, 0));

            // jump to next M block
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, m * 4 * area.kernelsize.N, 0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, m * 4 * area.kernelsize.N, 0));

            // jump N loop
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x9, (n_loop_count - m_kernel.get_size()) / 4));
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

        return Unary::error_t::success;
    }

    mini_jit::generator::Unary::kernel_t mini_jit::generator::Unary::get_kernel() const {
        return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
    }
}  // namespace mini_jit::generator