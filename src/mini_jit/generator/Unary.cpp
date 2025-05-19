#include "Unary.h"

#include "../instructions/instructions.h"
#include "util.h"

namespace inst = mini_jit::instructions;

#include "../instructions/instructions.h"

namespace inst = mini_jit::instructions;

mini_jit::generator::Unary::error_t mini_jit::generator::Unary::generate(uint32_t m,
                                                                         uint32_t n,
                                                                         uint32_t trans_b,
                                                                         dtype_t dtype,
                                                                         ptype_t ptype) {
    // procedure call standard (store to stack)
    m_kernel.add_instr(0x6DBF27E8);
    m_kernel.add_instr(0x6DBF2FEA);
    m_kernel.add_instr(0x6DBF37EC);
    m_kernel.add_instr(0x6DBF3FEE);

    // procedure call standard (load from stack)
    m_kernel.add_instr(0x6CC13FEE);
    m_kernel.add_instr(0x6CC137EC);
    m_kernel.add_instr(0x6CC12FEA);
    m_kernel.add_instr(0x6CC127E8);

    // ret
    m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

    m_kernel.set_kernel();

    m_kernel.write("output_test_unary.bin");

    return mini_jit::generator::Unary::error_t::success;
}

mini_jit::generator::Unary::kernel_t mini_jit::generator::Unary::get_kernel() const {
    return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
}