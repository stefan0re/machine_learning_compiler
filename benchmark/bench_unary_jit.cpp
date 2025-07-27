#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../src/mini_jit/generator/Unary.h"
#include "../src/mini_jit/generator/Util.h"
#include "../src/mini_jit/include/gemm_ref.h"

using namespace mini_jit::generator;

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
        float* l_in_transposed = new float[l_m * l_n];
        for (size_t j = 0; j < l_n; j++) {
            for (size_t i = 0; i < l_m; i++) {
                l_in_transposed[l_n * i + j] = l_in[j * l_m + i];
            }
        }
        delete[] l_out_ref;
        l_out_ref = l_in_transposed;
    }

    Unary l_unary;
    l_unary.generate(l_m, l_n, Unary::dtype_t::fp32, i_type);

    Unary::kernel_t unary_kernel = l_unary.get_kernel();

    unary_kernel(l_in, l_out, l_m, l_n);

    double l_error = 0.0;
    for (size_t i = 0; i < l_m * l_n; i++) {
        double l_tmp = std::abs(l_out[i] - l_out_ref[i]);
        if (l_tmp > 0.0) {
            std::cout << "Error at [" << i << "]: " << l_tmp << " (" << l_out[i] << " - " << l_out_ref[i] << ")" << std::endl;
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

int main(int argc, char** argv) {
    Unary::ptype_t i_type = Unary::ptype_t::zero;
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <type>" << std::endl;
        std::cerr << "Types: 0 - zero, 1 - identity, 2 - relu, 3 - transpose" << std::endl;
        return 1;
    }

    benchmark_unary_jit(static_cast<Unary::ptype_t>(atoi(argv[3])), atoi(argv[1]), atoi(argv[2]));
}