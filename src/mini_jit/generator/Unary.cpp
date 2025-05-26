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

    int32_t Unary::gen_unary_zero(mini_jit::generator::Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;
        int32_t op_count = 0;
        // m_kernel.add_instr(0x4F030480);  // place 100

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;
        int rem = count % 4;

        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
                op_count++;
            }
        }

        for (int i = 0; i < rem; i++) {
            // load one element at a time (.s[N])
            m_kernel.add_instr(
                inst::InstGen::neon_movi_zero(
                    static_cast<inst::InstGen::simd_fp_t>(reg_count), true, false));
            op_count++;
        }

        return reg_count;
    }

    int32_t Unary::gen_unary_relu(mini_jit::generator::Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;
        int32_t op_count = 0;

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;
        int rem = count % 4;

        op_count++;

        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                reg_count++;
                op_count++;
            }
        }

        for (int i = 0; i < rem; i++) {
            // load one element at a time (.s[N])
            m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                               static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                               inst::InstGen::simd_fp_t::v31,
                                                               false));
            op_count++;
        }

        return reg_count;
    }

    Unary::error_t Unary::get_kernel_sizes(uint32_t m,
                                           uint32_t n,
                                           Util::KernelSizes& kernelsizes,
                                           bool only_square) {
        int32_t max_reg_space = 32;

        std::vector<Util::KernelSize> work_areas;
        Util::get_area_sizes(m, n, work_areas);

        // define weights for scoring
        double w_rl = 1.2;
        double w_mn = 0.45;

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
                    if ((n_temp != m_temp) && only_square) continue;

                    // get used registers
                    int32_t mxn = m_temp * n_temp;
                    int32_t A_regs = (mxn - (mxn % 4)) / 4 + ((mxn % 4 == 0) ? 0 : 1);
                    int32_t B_regs = (mxn - (mxn % 4)) / 4 + ((mxn % 4 == 0) ? 0 : 1);

                    int32_t used_reg_space = A_regs + B_regs;

                    if (max_reg_space >= used_reg_space && (area.M % m_temp == 0 && area.N % n_temp == 0)) {
                        // metrix for how much bigger m is compared to n
                        double n_greater_m_deficit = (double)n_temp / (double)m_temp;

                        // relative number of unused registers
                        double registers_left = (max_reg_space - used_reg_space) / (double)max_reg_space;

                        double score = w_rl * registers_left + w_mn * n_greater_m_deficit;

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

        // std::cout << "MainArea Kernel      M: " << kernelsizes.kernel1.M << ", N: " << kernelsizes.kernel1.N << "\n"
        //<< "RightArea Kernel     M: " << kernelsizes.kernel2.M << ", N: " << kernelsizes.kernel2.N << "\n"
        //<< "LowerArea Kernel     M: " << kernelsizes.kernel3.M << ", N: " << kernelsizes.kernel3.N << "\n"
        //<< "RemainderArea Kernel M: " << kernelsizes.kernel4.M << ", N: " << kernelsizes.kernel4.N << "\n"
        //<< std::endl;

        return Unary::error_t::success;
    }

    Unary::error_t Unary::generate(uint32_t m,
                                   uint32_t n,
                                   uint32_t trans_b,
                                   Unary::dtype_t dtype,
                                   Unary::ptype_t ptype) {
        // safely calculate number of iterations for main loop and number of rest elements
        uint64_t total = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
        uint32_t iterations = static_cast<uint32_t>((total - (total % 4)) / 16);
        uint32_t rest = static_cast<uint32_t>(total % 16);

        // procedure call standard (store to stack)
        m_kernel.add_instr(0x6DBF27E8);
        m_kernel.add_instr(0x6DBF2FEA);
        m_kernel.add_instr(0x6DBF37EC);
        m_kernel.add_instr(0x6DBF3FEE);

        //  Store pointers of A and B to x7, x8
        m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x7,
                                                            inst::InstGen::x0));
        m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x8,
                                                            inst::InstGen::x1));

        // shift leading dimensions to 4 bytes
        m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x2, inst::InstGen::x2, 2));
        m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x3, inst::InstGen::x3, 2));

        // move 0 to v31 for relu
        m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::simd_fp_t::v31, true, false));

        // generate main loop
        if (iterations > 0) {
            // set loop counter, if number of iterations too high for immediate use movk
            uint64_t value = iterations;
            uint16_t lo = value & 0xffff;
            uint16_t hi = (value >> 16) & 0xffff;

            m_kernel.add_instr(inst::InstGen::base_movz(inst::InstGen::x9, lo, 0));  // movz x9, lo
            if (hi != 0)
                m_kernel.add_instr(inst::InstGen::base_movk(inst::InstGen::x9, hi, 16));  // movk x9, hi, LSL #16

            // loop
            size_t loop_count = m_kernel.get_size();

            m_kernel.add_instr(
                inst::InstGen::base_sub_imm(
                    inst::InstGen::x9,
                    inst::InstGen::x9,
                    1,
                    0));

            m_kernel.add_instr(inst::InstGen::neon_ld1_multiple(inst::InstGen::v0,
                                                                inst::InstGen::x7,
                                                                inst::InstGen::ld1_opcode_t::four_regs,
                                                                inst::InstGen::ld1_t::S4));

            if (ptype == Unary::ptype_t::zero) {
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v0, true, false));
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v1, true, false));
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v2, true, false));
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v3, true, false));
                this->fops += 4;
            } else if (ptype == Unary::ptype_t::relu) {
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v0,
                                                                   inst::InstGen::v0,
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v1,
                                                                   inst::InstGen::v1,
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v2,
                                                                   inst::InstGen::v2,
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v3,
                                                                   inst::InstGen::v3,
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                this->fops += 4;
            }

            m_kernel.add_instr(inst::InstGen::neon_st1_multiple(inst::InstGen::v0,
                                                                inst::InstGen::x8,
                                                                inst::InstGen::ld1_opcode_t::four_regs,
                                                                inst::InstGen::ld1_t::S4));

            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4 * 16, 0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4 * 16, 0));

            // jump loop
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x9, (loop_count - m_kernel.get_size()) / 4));
        }

        // try to use ld1 with as many registers as possible for rest (rest in [0, 15])
        uint32_t next_bigger = rest;

        while (next_bigger % 4 != 0) {
            next_bigger--;
        }

        inst::InstGen::ld1_opcode_t num_regs;

        if (next_bigger == 12) {
            num_regs = inst::InstGen::ld1_opcode_t::three_regs;
        } else if (next_bigger == 8) {
            num_regs = inst::InstGen::ld1_opcode_t::two_regs;
        } else if (next_bigger == 4) {
            num_regs = inst::InstGen::ld1_opcode_t::one_regs;
        }

        if (next_bigger > 0) {
            m_kernel.add_instr(inst::InstGen::neon_ld1_multiple(inst::InstGen::v0,
                                                                inst::InstGen::x7,
                                                                num_regs,
                                                                inst::InstGen::ld1_t::S4));

            int32_t reg_count = 0;

            for (int i = 0; i < (int)(next_bigger / 4); i++) {
                if (ptype == Unary::ptype_t::zero) {
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
                } else if (ptype == Unary::ptype_t::relu) {
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                       static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    reg_count++;
                }
            }

            m_kernel.add_instr(inst::InstGen::neon_st1_multiple(inst::InstGen::v0,
                                                                inst::InstGen::x8,
                                                                num_regs,
                                                                inst::InstGen::ld1_t::S4));

            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4 * next_bigger, 0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4 * next_bigger, 0));
        }

        // final rest (in [0, 3]) with single ld1 statements per element
        int32_t reg_count = 0;
        rest = (uint32_t)std::abs((int)next_bigger - (int)rest);
        for (int i = 0; i < rest; i++) {
            m_kernel.add_instr(
                inst::InstGen::neon_ld1_no_offset(
                    static_cast<inst::InstGen::simd_fp_t>(i),
                    inst::InstGen::x7,
                    inst::InstGen::vector_count_t::vc1));

            if (ptype == Unary::ptype_t::zero) {
                m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
            } else if (ptype == Unary::ptype_t::relu) {
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                reg_count++;
            }
            m_kernel.add_instr(
                inst::InstGen::neon_st1_no_offset(
                    static_cast<inst::InstGen::simd_fp_t>(i),
                    inst::InstGen::x8,
                    inst::InstGen::vector_count_t::vc1));

            // advance the base pointer by 1 elements
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4, 0));
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4, 0));
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