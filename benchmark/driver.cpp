#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "../src/mini_jit/generator/Unary.h"
#include "../test/test_utils.h"

int benchmark_unary(uint32_t m, uint32_t n, int iterations, mini_jit::generator::Unary::ptype_t ptype) {
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Benchmarking Unary: ";

    double ops_per_call;
    if (ptype == mini_jit::generator::Unary::ptype_t::zero) {
        std::cout << "Zero " << std::endl;
    } else if (ptype == mini_jit::generator::Unary::ptype_t::relu) {
        std::cout << "Relu " << std::endl;
    } else if (ptype == mini_jit::generator::Unary::ptype_t::identity) {
        std::cout << "Identity " << std::endl;
    }

    size_t size = m * n;
    float* a = new float[size];
    float* b = new float[size];
    std::chrono::_V2::system_clock::time_point start, end;
    bool is_correct = true;

    test_utils::generate_matrix(m, n, a);
    test_utils::generate_matrix(m, n, b);
    std::cout << "Matrix dimensions of " << m << "x" << n << std::endl;

    mini_jit::generator::Unary unary;

    unary.generate(m, n, 0, mini_jit::generator::Unary::dtype_t::fp32, ptype);
    mini_jit::generator::Unary::kernel_t kernel = unary.get_kernel();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernel(a, b, m, m);
    }
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = ((double)iterations * (double)((unary.fops * 4 * 4 * 16))) / duration;  // 192 flops in one iter

    std::cout << "\nIterations:\t" << iterations << " times" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GFLOPS\n"
              << std::endl;

    delete[] a;
    delete[] b;

    return 1;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    benchmark_unary(50, 50, 11000000, mini_jit::generator::Unary::ptype_t::zero);
    benchmark_unary(64, 64, 10000000, mini_jit::generator::Unary::ptype_t::zero);
    benchmark_unary(512, 512, 9000000, mini_jit::generator::Unary::ptype_t::zero);
    benchmark_unary(2048, 2048, 10000, mini_jit::generator::Unary::ptype_t::zero);
    benchmark_unary(50, 50, 2500000, mini_jit::generator::Unary::ptype_t::relu);
    benchmark_unary(64, 64, 2500000, mini_jit::generator::Unary::ptype_t::relu);
    benchmark_unary(512, 512, 100000, mini_jit::generator::Unary::ptype_t::relu);
    benchmark_unary(2048, 2048, 10000, mini_jit::generator::Unary::ptype_t::relu);

    return 0;
}
