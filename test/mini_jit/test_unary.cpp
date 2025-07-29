#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/mini_jit/generator/Unary.h"
#include "../../src/mini_jit/generator/Util.h"
#include "../test_utils/test_utils.h"

using namespace mini_jit::generator;

void transpose_ref(int i_m,
                   int i_n,
                   int i_ldi,
                   int i_ldo,
                   const float* i_matrix,
                   float* o_matrix) {
    for (size_t l_n = 0; l_n < i_n; l_n++) {
        for (size_t l_m = 0; l_m < i_m; l_m++) {
            o_matrix[l_m * i_ldo + l_n] = i_matrix[l_n * i_ldi + l_m];
        }
    }
}

using namespace mini_jit::generator;

TEST_CASE("MiniJit::Unary Tests Unary ZERO", "[MiniJit][UNARY]") {
    int sizes[1] = {32};
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
                l_error += l_tmp;
            }
        }
        std::cout << "Config: " << l_m << "x" << l_n << std::endl;
        std::cout << "Total Error: " << l_error << std::endl;

        delete[] l_in;
        delete[] l_out;
        delete[] l_out_ref;
    }
}

TEST_CASE("MiniJit::Unary Tests Unary TRANSPOSE", "[MiniJit][UNARY]") {
    int sizes[1] = {32};
    for (size_t i = 0; i < 1; i++) {
        int i_m = sizes[i];
        int i_n = sizes[i];
        int i_ldi = i_m;
        int i_ldo = i_n;
        std::cout << "Running Transpose with: " << std::endl;
        std::cout << "M = " << i_m << ", N = " << i_n << ", LDI = " << i_ldi << ", LDO = " << i_ldo << std::endl;
        srand48(time(NULL));

        float* l_in = new float[i_ldi * i_n];
        float* l_out = new float[i_m * i_ldo];
        float* l_out_ref = new float[i_m * i_ldo];

        int count = 1;
        for (size_t l_n = 0; l_n < i_n; l_n++) {
            for (size_t l_m = 0; l_m < i_ldi; l_m++) {
                if (l_m < i_m) {
                    l_in[l_n * i_ldi + l_m] = (float)count++;
                } else {
                    l_in[l_n * i_ldi + l_m] = 0.0f;
                }
            }
        }

        transpose_ref(i_m, i_n, i_ldi, i_ldo, l_in, l_out_ref);

        Unary l_unary;
        l_unary.generate(i_m, i_n, Unary::dtype_t::fp32, Unary::ptype_t::trans);

        Unary::kernel_t unary_kernel = l_unary.get_kernel();

        unary_kernel(l_in, l_out, i_ldi, i_ldo);

        // compare results
        double l_error = 0.0;
        for (size_t i = 0; i < i_m * i_ldo; i++) {
            double l_tmp = std::abs(l_out[i] - l_out_ref[i]);
            if (l_tmp > 1e-5) {
                l_error += l_tmp;
            }
        }
        std::cout << "Config: " << i_m << "x" << i_n << " lda = " << i_ldi << ", ldo = " << i_ldo << std::endl;
        std::cout << "Total Error: " << l_error << std::endl;
        REQUIRE(l_error < 1e-5);
    }
}

/* TODO add tests with using a new kernel */