#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../src/mini_jit/generator/Brgemm.h"
#include "../src/mini_jit/generator/Unary.h"
#include "../src/mini_jit/generator/Util.h"
#include "../src/mini_jit/include/gemm_ref.h"

using namespace mini_jit::generator;

void benchmark_brgemm() {
    int64_t iterations = 0;
    double mean_gflops = 0.0;
    int br = 16;
    int k_sizes[5] = {1, 16, 32, 64, 128};
    for (size_t n = 1; n < 65; n++) {
        for (size_t m = 1; m < 65; m++) {
            for (int k : k_sizes) {
                iterations++;
                mini_jit::generator::Brgemm l_brgemm;
                l_brgemm.generate(m, n, k, br, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

                float *l_a = (float *)malloc(br * m * k * sizeof(float));
                float *l_b = (float *)malloc(br * k * n * sizeof(float));
                float *l_c_jit = (float *)malloc(m * n * sizeof(float));
                float *l_c_ref = (float *)malloc(m * n * sizeof(float));

                srand48(time(NULL));

                for (int i = 0; i < br * m * k; i++) {
                    l_a[i] = (float)drand48();
                }
                for (int i = 0; i < br * k * n; i++) {
                    l_b[i] = (float)drand48();
                }
                for (int i = 0; i < m * n; i++) {
                    l_c_jit[i] = (float)drand48();
                    l_c_ref[i] = l_c_jit[i];
                }

                brgemm_ref(l_a, l_b, l_c_ref,
                           m, n, k, br,
                           m, k, m, m * k, n * k);
                mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
                l_kernel(l_a, l_b, l_c_jit,
                         m, k, m,
                         m * k, k * n);

                double l_error = 0.0;
                for (size_t i = 0; i < m * n; i++) {
                    l_error += std::abs(l_c_jit[i] - l_c_ref[i]);
                }
                if (l_error > 1e-4) {
                    std::cerr << "Error at config:" << m << "," << n << "," << k << std::endl;
                    return;
                }

                // get iteration by testing a few iterations
                auto l_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10; i++) {
                    l_kernel(l_a, l_b, l_c_jit,
                             m, k, m,
                             m * k, k * n);
                }
                auto l_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> l_duration = l_end - l_start;
                int64_t iteration = 20.0 / l_duration.count();

                l_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iteration; i++) {
                    l_kernel(l_a, l_b, l_c_jit,
                             m, k, m,
                             m * k, k * n);
                }
                l_end = std::chrono::high_resolution_clock::now();
                l_duration = l_end - l_start;
                double gflops = (2.0 * m * n * k) * iteration;
                gflops /= l_duration.count();
                gflops /= 1e9;
                std::cout << "  Duration: " << l_duration.count() << std::endl;
                std::cout << "  GFLOPS: " << gflops << std::endl;
                mean_gflops += gflops;

                std::cout << "CSV:" << m << "," << n << "," << k << "," << br << ","
                          << m << "," << k << "," << m << "," << "0,0,"
                          << l_duration.count() << "," << gflops << std::endl;
                std::cout << "--------------------------" << std::endl;

                free(l_a);
                free(l_b);
                free(l_c_jit);
                free(l_c_ref);
            }
        }
    }
    mean_gflops /= iterations;
    std::cout << "************************************" << std::endl;
    std::cout << "Arithmetic mean performance: " << mean_gflops << std::endl;
    std::cout << "************************************" << std::endl;
}

int main() {
    benchmark_brgemm();
}