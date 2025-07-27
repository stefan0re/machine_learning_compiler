#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/mini_jit/generator/Unary.h"
#include "../../src/mini_jit/generator/Util.h"
#include "../test_utils/test_utils.h"

using namespace mini_jit::generator;

TEST_CASE("MiniJit::Unary Tests Unary ZERO", "[MiniJit][UNARY]") {
    int sizes[8] = {32, 50, 64, 100, 222, 243, 512, 2048};
    for (int size : sizes) {
        int l_m = size;
        int l_n = size;

        std::cout << "Running Unary Benchmark with: M = " << l_m << ", N = " << l_n << std::endl;

        srand48(l_m * l_n);

        float* l_in = new float[l_m * l_n];
        float* l_out = new float[l_m * l_n];
        float* l_out_ref = new float[l_m * l_n];

        for (size_t i = 0; i < l_m * l_n; i++) {
            l_in[i] = (float)drand48() * 10 - 5;
        }

        for (size_t i = 0; i < l_m * l_n; i++) {
            l_out[i] = (float)drand48() * 10 - 5;
            l_out_ref[i] = 0.0;
        }

        Unary l_unary;
        l_unary.generate(l_m, l_n, Unary::dtype_t::fp32, Unary::ptype_t::zero);

        Unary::kernel_t zero = l_unary.get_kernel();

        zero(l_in, l_out, l_m, l_n);

        double l_error = 0.0;
        for (size_t i = 0; i < l_m * l_n; i++) {
            double l_tmp = std::abs(l_out[i] - l_out_ref[i]);
            if (l_tmp > 0.0) {
                std::cout << "Error at [" << i << "]: " << l_tmp << std::endl;
                l_error += l_tmp;
            }
        }
        std::cout << "Total Error: " << l_error << std::endl;

        delete[] l_in;
        delete[] l_out;
        delete[] l_out_ref;
    }
}
