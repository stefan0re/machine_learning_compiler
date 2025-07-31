#include <chrono>
#include <iostream>
#include <string>

#include "../src/einsum/trees/einsum_trees.h"
#include "../src/mini_jit/include/gemm_ref.h"
#include "../src/tensor/tensor.h"

using namespace einsum::trees;

#define BATCH_SIZE 1

/** Running Model from einsum tree:
 * string: [[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]
 *
 * 0 => BATCH_SIZE
 * 1 => 4
 * 2 => 64
 * 3 => 16
 * 4 => 3

 * GEMM 0: M = BATCH_SIZE, N = 64, K = 4
 * GEMM 1: M = BATCH_SIZE, N = 16, K = 64
 * GEMM 2: M = BATCH_SIZE, N = 3, K = 16
 */
void test_model() {
    std::string str_repr = "[[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]";
    EinsumTree model_tree = EinsumTree(str_repr, {BATCH_SIZE, 4, 64, 16, 3}, true);
    model_tree.optimize();
    model_tree.lower();

    std::cout << "Running model ..." << std::endl;
    std::cout << "*******************************************" << std::endl;
    std::cout << "String representation: " << str_repr << std::endl;
    std::cout << "Sizes: (" << BATCH_SIZE << ", 4, 64, 16, 3)" << std::endl;
    std::cout << "*******************************************" << std::endl;
    model_tree.print();

    Tensor W1 = Tensor(64, 4);
    Tensor b1 = Tensor(64);
    Tensor W2 = Tensor(64, 16);
    Tensor b2 = Tensor(16);
    Tensor W3 = Tensor(16, 3);
    Tensor b3 = Tensor(3);

    // input
    Tensor input = Tensor(4, BATCH_SIZE);
    srand48(time(NULL));
    // init tensors with random values
    for (size_t i = 0; i < 4 * BATCH_SIZE; i++) {
        input.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 4 * BATCH_SIZE; i++) {
        W1.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 64; i++) {
        b1.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 64 * 16; i++) {
        W2.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 16; i++) {
        b2.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 16 * 3; i++) {
        W3.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 3; i++) {
        b3.data[i] = (float)drand48() * 10 - 5;
    }

    std::vector<void*> model_inputs = {static_cast<void*>(input.data),
                                       static_cast<void*>(W1.data),
                                       static_cast<void*>(W2.data),
                                       static_cast<void*>(W3.data)};
    std::vector<void*> model_bias = {static_cast<void*>(b3.data),
                                     static_cast<void*>(b2.data),
                                     static_cast<void*>(b1.data)};

    float* model_output = new float[3 * BATCH_SIZE];
    for (size_t i = 0; i < 3 * BATCH_SIZE; i++) {
        model_output[i] = 0.0f;
    }

    // benchmark execution
    int64_t reps = 1;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < reps; i++)
        model_tree.execute(model_inputs, model_bias, model_output);
    auto end = std::chrono::high_resolution_clock::now();

    double flops = (2 * BATCH_SIZE * 4 * 64 * 16 * 3) - (BATCH_SIZE * 3);
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;
    double gflops = (flops / duration.count()) * 1e-6;
    gflops *= reps;
    std::cout << "  GFLOPS: " << gflops << std::endl;
}

int main() {
    test_model();

    return 0;
}