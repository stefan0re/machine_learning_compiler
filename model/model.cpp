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

void model_ref(std::vector<void*> in_tensors,
               std::vector<void*> biases,
               float* out) {
    float* result_gemm_0 = new float[BATCH_SIZE * 64];
    float* result_gemm_1 = new float[BATCH_SIZE * 16];

    for (size_t i = 0; i < BATCH_SIZE * 64; i++) {
        result_gemm_0[i] = 0;
    }
    for (size_t i = 0; i < BATCH_SIZE * 16; i++) {
        result_gemm_1[i] = 0;
    }

    // GEMM 0: M = 16, N = 64, K = 4
    gemm_ref(static_cast<float*>(in_tensors[0]), static_cast<float*>(in_tensors[1]), result_gemm_0, BATCH_SIZE, 64, 4, BATCH_SIZE, 4, BATCH_SIZE);
    std::cout << result_gemm_0[0] << std::endl;

    // Add bias1 to result_gemm_0
    float* bias1 = static_cast<float*>(biases[0]);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        for (size_t j = 0; j < 64; j++) {
            result_gemm_0[i * 64 + j] += bias1[j];
        }
    }

    // Apply ReLU activation
    for (size_t i = 0; i < BATCH_SIZE * 64; i++) {
        if (result_gemm_0[i] < 0) {
            result_gemm_0[i] = 0.0;
        }
    }

    // GEMM 1: M = 16, N = 16, K = 64
    gemm_ref(result_gemm_0, static_cast<float*>(in_tensors[2]), result_gemm_1, BATCH_SIZE, 16, 64, BATCH_SIZE, 64, BATCH_SIZE);
    std::cout << result_gemm_1[0] << std::endl;

    // Add bias2 to result_gemm_1
    float* bias2 = static_cast<float*>(biases[1]);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        for (size_t j = 0; j < 16; j++) {
            result_gemm_1[i * 16 + j] += bias2[j];
        }
    }

    // Apply ReLU activation
    for (size_t i = 0; i < BATCH_SIZE * 16; i++) {
        if (result_gemm_1[i] < 0) {
            result_gemm_1[i] = 0.0;
        }
    }

    // GEMM 2: M = 16, N = 3, K = 16
    gemm_ref(result_gemm_1, static_cast<float*>(in_tensors[3]), out, BATCH_SIZE, 3, 16, BATCH_SIZE, 16, BATCH_SIZE);
    std::cout << out[0] << std::endl;

    // Add bias3 to out
    float* bias3 = static_cast<float*>(biases[2]);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        for (size_t j = 0; j < 3; j++) {
            out[i * 3 + j] += bias3[j];
        }
    }
}

int main() {
    int in_size = 4;
    std::string str_repr = "[[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]";

    std::vector<Tensor> in_tensors = Tensor::from_torchpp("../python/model.torchpp", in_size);

    Tensor dataset = Tensor::from_csv("../python/iris.csv");
    std::cout << "Dataset: \n"
              << dataset.info_str() << std::endl;

    std::vector<void*> data_and_weights = {
        dataset.data,  // input data
    };
    std::vector<void*> biases;

    uint32_t index = 0;
    for (auto& t : in_tensors) {
        index++;

        if (t.id.size() == 2) {
            // this is a weight matrix
            std::cout << "Weight Tensor " << index << ": " << std::endl;
            data_and_weights.push_back(static_cast<void*>(t.data));
        } else if (t.id.size() == 1) {
            // this is a bias vector
            std::cout << "Bias Tensor " << index << ": " << std::endl;
            biases.push_back(static_cast<void*>(t.data));
        } else {
            std::cout << "Unknown Tensor type for Tensor " << index << ": " << std::endl;
        }
        t.info();
    }

    float* l_out = new float[BATCH_SIZE * 3];
    float* l_out_ref = new float[BATCH_SIZE * 3];

    srand48(time(NULL));

    for (size_t i = 0; i < BATCH_SIZE * 3; i++) {
        l_out[i] = 0.0f;
        l_out_ref[i] = 0.0f;
    }

    model_ref(data_and_weights, biases, l_out_ref);

    EinsumTree tree = EinsumTree(str_repr, {BATCH_SIZE, 4, 64, 16, 3}, true);
    tree.optimize();
    tree.lower();
    tree.print();

    tree.execute(data_and_weights, biases, l_out);
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

    /*auto tp0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < reps; i++) {
        tree.execute(data_and_weights, biases, l_out);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = tp1 - tp0;
    double time = duration.count();
    double gflops = (flops * reps) / (time * 1e9);
    std::cout << "Execution time for 1000 iterations: " << time << " seconds" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;*/

    std::cout << "Number of errors: " << count_error << std::endl;
    return EXIT_SUCCESS;
}