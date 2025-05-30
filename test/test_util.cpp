#include "../src/mini_jit/generator/util.h"

int main(int argc, char const *argv[]) {
    mini_jit::generator::Util util;

    mini_jit::generator::Util::KernelSize kernelSize;
    kernelSize.M = 5;
    kernelSize.N = 6;
    int offset_k = 64;

    int32_t used_vector_reg_count = mini_jit::generator::Util::gen_c_load(kernelSize);
    mini_jit::generator::Util::gen_microkernel(kernelSize, used_vector_reg_count);
    mini_jit::generator::Util::gen_c_store(kernelSize);

    return 0;
}