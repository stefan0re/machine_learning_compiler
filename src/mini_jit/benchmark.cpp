#include <iostream>
#include <vector>
#include <chrono>

#include "generator/Brgemm.h"
#include "include/gemm_ref.h"

int main( int argc, char *argv[] ) {
    std::cout << "mini_jit benchmark" << std::endl;
    std::cout << "===================" << std::endl;

    int64_t m = 16;
    int64_t n = 6;
    int64_t k = (argc > 1) ? atoi(argv[1]) : 64;

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    const int64_t br_stride_a = m * k;
    const int64_t br_stride_b = k * n;

    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

    // generate random A B and C 
    srand48( time(NULL) );

    // initialize matrix
    float * l_a = (float *) malloc( lda * k * sizeof(float));
    float * l_b = (float *) malloc( ldb * n * sizeof(float));
    float * l_c_1 = (float *) malloc( ldc * n * sizeof(float));
    float * l_c_2 = (float *) malloc( ldc * n * sizeof(float));
    
    for( int i = 0; i < lda * k; i++ ) {
        l_a[i] = (float)drand48();
    }
    for( int i = 0; i < ldb * n; i++ ) {
        l_b[i] = (float)drand48();
    }
    for( int i = 0; i < ldc * n; i++ ) {
        l_c_1[i] = (float)drand48();
        l_c_2[i] = l_c_1[i];
    }

    // compute reference
    gemm_ref(l_a, l_b, l_c_1,
             m, n, k,
             lda, ldb, ldc);

    // compute jiter
    mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

    l_kernel(l_a, l_b, l_c_2,
             lda, ldb, ldc,
             br_stride_a, br_stride_b);

    // compare results
    double l_diff = 0.0;
    for( int i = 0; i < ldc * n; i++ ) {
        l_diff += fabs(l_c_1[i] - l_c_2[i]);
        if( fabs(l_c_1[i] - l_c_2[i]) > 0.0001 ){
            std::cout << "Error: " << l_c_1[i] << " != " << l_c_2[i] << std::endl;
        }
    }

    // Benchmark Kernel
    uint64_t l_interations = 50;

    // define l_iterations
    auto l_st = std::chrono::high_resolution_clock::now();
    for( uint32_t i = 0; i < l_interations; i++ ) {
        l_kernel(l_a, l_b, l_c_2,
                 lda, ldb, ldc,
                 br_stride_a, br_stride_b);
    }
    auto l_et = std::chrono::high_resolution_clock::now();

    double l_duration = std::chrono::duration<double>(l_et - l_st).count();
    l_interations = (uint64_t)(500.0 / l_duration);

    // warm up
    l_kernel( l_a, l_b, l_c_2,
              lda, ldb, ldc,
              br_stride_a, br_stride_b);
    auto l_start = std::chrono::high_resolution_clock::now();
    for( uint32_t i = 0; i < l_interations; i++ ) {
        l_kernel(l_a, l_b, l_c_2,
                 lda, ldb, ldc,
                 br_stride_a, br_stride_b);
    }
    auto l_end = std::chrono::high_resolution_clock::now();


    l_duration = std::chrono::duration<double>(l_end - l_start).count();
    double l_gflops = 2.0 * m * n * k;
    l_gflops *= l_interations;
    l_gflops /= l_duration;
    l_gflops /= 1e9;
    


    std::cout << "Dimensions: M = " << m << ", N = " << n << ", K = " << k << std::endl;
    std::cout << "Leading dims: A = " << lda << ", B = " << ldb << ", C = " << ldc << std::endl;
    std::cout << "Brgemm stride: A = " << br_stride_a << ", B = " << br_stride_b << std::endl;
    std::cout << "Iterations: " << l_interations << std::endl;
    std::cout << "Diff: " << l_diff << std::endl;
    std::cout << "GFLOPS: " << l_gflops << std::endl;
    std::cout << "Duration: " << l_duration << "s" << std::endl;
    std::cout << "===================" << std::endl;
    
    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);


}