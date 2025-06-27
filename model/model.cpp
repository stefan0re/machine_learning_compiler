#include <chrono>
#include <cstdlib>
#include <iostream>

#include "../src/einsum/trees/einsum_trees.h"
#include "../src/mini_jit/include/gemm_ref.h"
#include "../src/tensor/tensor.h"

/** Running Model from einsum tree:
 * string: [[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]
 *
 * 0 => 16
 * 1 => 4
 * 2 => 64
 * 3 => 16
 * 4 => 3
 */

using namespace einsum::trees;
/**
 * GEMM 0: M = 16/1, N = 64, K = 4
 * GEMM 1: M = 16, N = 16, K = 64
 * GEMM 2: M = 16, N = 3, K = 16
 */

#define BATCH_SIZE 16

void model_ref(float* in_0,
               float* in_1,
               float* in_2,
               float* in_3,
               float* out) {
    float* result_gemm_0 = new float[BATCH_SIZE * 64];
    float* result_gemm_1 = new float[BATCH_SIZE * 16];

    for (size_t i = 0; i < BATCH_SIZE * 64; i++) {
        result_gemm_0[i] = 0;
    }
    for (size_t i = 0; i < BATCH_SIZE * 16; i++) {
        result_gemm_1[i] = 0;
    }

    gemm_ref(in_0, in_1, result_gemm_0, BATCH_SIZE, 64, 4, BATCH_SIZE, 4, BATCH_SIZE);

    for (size_t i = 0; i < BATCH_SIZE * 64; i++) {
        if (result_gemm_0[i] < 0) {
            result_gemm_0[i] = 0.0;
        }
    }

    gemm_ref(result_gemm_0, in_2, result_gemm_1, BATCH_SIZE, 16, 64, BATCH_SIZE, 64, BATCH_SIZE);

    for (size_t i = 0; i < BATCH_SIZE * 16; i++) {
        if (result_gemm_1[i] < 0) {
            result_gemm_1[i] = 0.0;
        }
    }
    gemm_ref(result_gemm_1, in_3, out, BATCH_SIZE, 3, 16, BATCH_SIZE, 16, BATCH_SIZE);
}

int main() {
    std::string str_repr = "[[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]";

    float* l_in0 = new float[BATCH_SIZE * 4];
    float* l_in1 = new float[4 * 64];
    float* l_in2 = new float[64 * 16];
    float* l_in3 = new float[16 * 3];
    float* l_out = new float[BATCH_SIZE * 3];
    float* l_out_ref = new float[BATCH_SIZE * 3];

    srand48(time(NULL));

    for (size_t i = 0; i < BATCH_SIZE * 4; i++) {
        l_in0[i] = (float)(drand48() * 10.0f) - 5.0f;
    }
    for (size_t i = 0; i < 4 * 64; i++) {
        l_in1[i] = (float)(drand48() * 10.0f) - 5.0f;
    }
    for (size_t i = 0; i < 64 * 16; i++) {
        l_in2[i] = (float)(drand48() * 10.0f) - 5.0f;
    }
    for (size_t i = 0; i < 16 * 3; i++) {
        l_in3[i] = (float)(drand48() * 10.0f) - 5.0f;
    }
    for (size_t i = 0; i < BATCH_SIZE * 3; i++) {
        l_out[i] = 0.0f;
        l_out_ref[i] = 0.0f;
    }

    model_ref(l_in0, l_in1, l_in2, l_in3, l_out_ref);

    EinsumTree tree = EinsumTree(str_repr, {BATCH_SIZE, 4, 64, 16, 3});
    tree.optimize();
    tree.print();
    tree.lower();

    tree.execute({l_in0, l_in1, l_in2, l_in3}, l_out);

    // check if output is correct
    double error = 0.0;
    size_t count_error = 0;
    for (size_t i = 0; i < BATCH_SIZE * 3; i++) {
        error += fabs(l_out[i] - l_out_ref[i]);
        if (fabs(l_out[i] - l_out_ref[i]) > 1e-5) {
            std::cout << "Error at index " << i << ": " << l_out[i] << " != " << l_out_ref[i] << std::endl;
            count_error++;
        }
    }
    std::cout << "Error: " << error << std::endl;

    double flops = (2.0 * BATCH_SIZE * 4 * 64 * 16 * 3) - 3 * BATCH_SIZE;

    size_t reps = 10000;

    auto tp0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < reps; i++) {
        tree.execute({l_in0, l_in1, l_in2, l_in3}, l_out);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = tp1 - tp0;
    double time = duration.count();
    double gflops = (flops * reps) / (time * 1e9);
    std::cout << "Execution time for 1000 iterations: " << time << " seconds" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;

    return EXIT_SUCCESS;
}