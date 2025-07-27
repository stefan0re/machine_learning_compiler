#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../src/mini_jit/generator/Unary.h"
#include "../src/mini_jit/generator/Util.h"
#include "../src/mini_jit/include/gemm_ref.h"

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

void benchmark_unary_jit(Unary::ptype_t i_type,
                         int i_m,
                         int i_n) {
    int l_m = i_m;
    int l_n = i_n;

    std::cout << "Running Unary " << static_cast<int>(i_type) << " Benchmark with: M = " << l_m << ", N = " << l_n << std::endl;

    srand48(2);

    float* l_in = new float[l_m * l_n];
    float* l_out = new float[l_m * l_n];
    float* l_out_ref = new float[l_m * l_n];

    for (size_t i = 0; i < l_m * l_n; i++) {
        l_in[i] = (float)i + 1;  // drand48() * 10 - 5;
    }

    for (size_t i = 0; i < l_m * l_n; i++) {
        l_out[i] = (float)i + 1;  // drand48() * 10 - 5;
    }

    for (size_t i = 0; i < l_m * l_n; i++) {
        if (i_type == Unary::ptype_t::zero) {
            l_out_ref[i] = 0.0f;
        } else if (i_type == Unary::ptype_t::relu) {
            l_out_ref[i] = std::max(0.0f, l_in[i]);
        } else if (i_type == Unary::ptype_t::identity) {
            l_out_ref[i] = l_in[i];
        } else {
            break;
        }
    }

    if (i_type == Unary::ptype_t::trans) {
        transpose_ref(i_m, i_n, i_m, i_n, l_in, l_out_ref);
    }

    Unary l_unary;
    l_unary.generate(l_m, l_n, Unary::dtype_t::fp32, i_type);

    Unary::kernel_t unary_kernel = l_unary.get_kernel();

    unary_kernel(l_in, l_out, l_m, l_n);

    double l_error = 0.0;
    for (size_t i = 0; i < l_m * l_n; i++) {
        double l_tmp = std::abs(l_out[i] - l_out_ref[i]);
        if (l_tmp > 0.0) {
            // std::cout << "Error at [" << i << "]: " << l_tmp << " (" << l_out[i] << " - " << l_out_ref[i] << ")" << std::endl;
            l_error += l_tmp;
        }
    }
    std::cout << "Total Error: " << l_error << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 20; i++) {
        unary_kernel(l_in, l_out, l_m, l_n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    long iterations = 100.0 / duration.count();

    // measure GiB/s of the kernel
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; i++) {
        unary_kernel(l_in, l_out, l_m, l_n);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    double GiB = (static_cast<double>(l_m) * l_n * sizeof(float)) * 2 * iterations / (1024.0 * 1024.0 * 1024.0);
    double gibs_per_sec = GiB / duration.count();
    std::cout << "Performance: " << gibs_per_sec << " GiB/s" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    delete[] l_in;
    delete[] l_out;
    delete[] l_out_ref;
}

void test_transpose(int i_m,
                    int i_n,
                    int i_ldi,
                    int i_ldo) {
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
            // std::cout << "Error at [" << i << "]: " << l_tmp << " (" << l_out[i] << " - " << l_out_ref[i] << ")" << std::endl;
            l_error += l_tmp;
        }
    }
    std::cout << "Total Error: " << l_error << std::endl;
}

int main(int argc, char** argv) {
    Unary::ptype_t i_type = Unary::ptype_t::zero;
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <type>" << std::endl;
        std::cerr << "Types: 0 - zero, 1 - identity, 2 - relu, 3 - transpose" << std::endl;
        std::cerr << "Or: " << argv[0] << " <m> <n> <type> <ldi> <ldo> for transpose" << std::endl;
        return 1;
    }
    if (argc == 4) {
        benchmark_unary_jit(static_cast<Unary::ptype_t>(atoi(argv[3])), atoi(argv[1]), atoi(argv[2]));
    } else if (argc == 5) {
        int l_m = atoi(argv[1]);
        int l_n = atoi(argv[2]);
        int l_ldi = atoi(argv[3]);
        int l_ldo = atoi(argv[4]);
        if (l_ldi < l_m || l_ldo < l_n) {
            std::cerr << "LDI must be >= M and LDO must be >= N for transpose." << std::endl;
            return 1;
        }
        test_transpose(l_m, l_n, l_ldi, l_ldo);
    } else {
        std::cerr << "Invalid number of arguments." << std::endl;
        return 1;
    }
}