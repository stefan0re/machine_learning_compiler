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

    void Unary::gen_transpose_micro( uint32_t i_m,
                                     uint32_t i_n) {
        // ldr
        for( size_t i = 0; i < 4; i++){
            m_kernel.add_instr(inst::InstGen::neon_ld1_no_offset( static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                                  inst::InstGen::x0,
                                                                  inst::InstGen::vector_count_t::vc1));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                        inst::InstGen::x0,
                                                                        inst::InstGen::x2,
                                                                        0,
                                                                        0));
        }
        /* first part */
        // trn 
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v4, inst::InstGen::v0, inst::InstGen::v1, 1));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v5, inst::InstGen::v0, inst::InstGen::v1, 2));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v6, inst::InstGen::v2, inst::InstGen::v3, 1));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v7, inst::InstGen::v2, inst::InstGen::v3, 2));

        // zip
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v8, inst::InstGen::v4, inst::InstGen::v6, 1));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v9, inst::InstGen::v5, inst::InstGen::v7, 1));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v10, inst::InstGen::v4, inst::InstGen::v6, 2));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v11, inst::InstGen::v5, inst::InstGen::v7, 2));



        // str
        for( size_t i = 0; i < 4; i++){
            m_kernel.add_instr(inst::InstGen::neon_st1_no_offset(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v8 + i),
                                                                        inst::InstGen::x1,
                                                                        inst::InstGen::vector_count_t::vc1));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                        inst::InstGen::x1,
                                                                        inst::InstGen::x3,
                                                                        0,
                                                                        0));
        }
    }

    void Unary::gen_transpose( uint32_t i_m,
                               uint32_t i_n){
        /* get blocking */
        uint32_t m_blocks_full = i_m / 4;
        uint32_t m_blocks_reminder = i_m % 4;

        uint32_t n_blocks_full = i_n / 4;
        uint32_t n_blocks_reminder = i_n % 4;

        // write restore size to x5 for A and x6 for B
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x5, 4, 0));
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x6, i_m, 0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg( inst::InstGen::x5,
                                                        inst::InstGen::x5,
                                                        inst::InstGen::x2 ));
        m_kernel.add_instr(inst::InstGen::base_mul_reg( inst::InstGen::x6,
                                                        inst::InstGen::x6,
                                                        inst::InstGen::x3 ));   


        for( uint32_t l_n = 0; l_n < n_blocks_full; l_n++){
            for( uint32_t l_m = 0; l_m < m_blocks_full; l_m++ ){
                gen_transpose_micro( 4, 4);
                if( l_m < (m_blocks_full - 1)){
                    m_kernel.add_instr(inst::InstGen::base_sub_shifted_register( inst::InstGen::x0,
                                                                                inst::InstGen::x0,
                                                                                inst::InstGen::x5,
                                                                                0,
                                                                                0));
                    
                    m_kernel.add_instr(inst::InstGen::base_add_imm( inst::InstGen::x0,
                                                                    inst::InstGen::x0,
                                                                    4 * 4,
                                                                    0));
                }
            }
            // adjust a and b pointer
            m_kernel.add_instr( inst::InstGen::base_sub_imm( inst::InstGen::x0,
                                                             inst::InstGen::x0,
                                                             4 * 4 * (m_blocks_full - 1),
                                                             0));
        
            
            m_kernel.add_instr(inst::InstGen::base_sub_shifted_register( inst::InstGen::x1,
                                                                         inst::InstGen::x1,
                                                                         inst::InstGen::x6,
                                                                         0,
                                                                         0));
            
            m_kernel.add_instr( inst::InstGen::base_add_imm( inst::InstGen::x1,
                                                             inst::InstGen::x1,
                                                             4 * 4,
                                                             0));
        }
    }

    void Unary::gen_zero( ){
        for( uint32_t i = 0; i < 32; i ++) {
            m_kernel.add_instr( inst::InstGen::neon_eor( static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                         static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                         static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i)));
        }
    }
    // assuming that vr 31 is not used by C accumulator
    void Unary::gen_relu(){
        m_kernel.add_instr( inst::InstGen::neon_eor( inst::InstGen::v31,
                                                     inst::InstGen::v31,
                                                     inst::InstGen::v31));
        for( uint32_t i = 0; i < 31; i ++) {
            m_kernel.add_instr( inst::InstGen::neon_fmax_vector( static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                                 static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v31 ),
                                                                 static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                                 false));
        }
    }
    

    Unary::error_t Unary::generate(uint32_t m,
                                   uint32_t n,
                                   Unary::dtype_t dtype,
                                   Unary::ptype_t ptype) {

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
        m_kernel.add_instr(0xd37ef442);
        m_kernel.add_instr(0xd37ef463);

        if( ptype == Unary::ptype_t::trans ){
            gen_transpose( m,
                       n);
        } else if( ptype == Unary::ptype_t::zero ){
             Util::KernelSize l_kernelsize;
            l_kernelsize.M = m;
            l_kernelsize.N = n;

            m_kernel.add_instr(inst::InstGen::base_mov_register( inst::InstGen::x5,
                                                                 inst::InstGen::x3));

            gen_zero();

            Util::generator_store_reg_block( m_kernel,
                                             l_kernelsize,
                                             inst::InstGen::x1);
        } else if( ptype == Unary::ptype_t::identity ){
            Util::KernelSize l_kernelsize;
            l_kernelsize.M = m;
            l_kernelsize.N = n;

            m_kernel.add_instr(inst::InstGen::base_mov_register( inst::InstGen::x5,
                                                                 inst::InstGen::x3));

            // load C
            Util::generator_load_reg_block( m_kernel,
                                            l_kernelsize,
                                            inst::InstGen::x0);

            // store C
            Util::generator_store_reg_block( m_kernel,
                                             l_kernelsize,
                                             inst::InstGen::x1);
        } else if( ptype == Unary::ptype_t::relu) {
            Util::KernelSize l_kernelsize;
            l_kernelsize.M = m;
            l_kernelsize.N = n;

            m_kernel.add_instr(inst::InstGen::base_mov_register( inst::InstGen::x5,
                                                                 inst::InstGen::x3));

            // load C
            Util::generator_load_reg_block( m_kernel,
                                            l_kernelsize,
                                            inst::InstGen::x0);

            gen_relu();

            // store C
            Util::generator_store_reg_block( m_kernel,
                                             l_kernelsize,
                                             inst::InstGen::x1);
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