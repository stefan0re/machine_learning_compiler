#include <chrono>
#include <iostream>
#include <string>

#include "../src/einsum/trees/einsum_trees.h"
#include "../src/mini_jit/include/gemm_ref.h"
#include "../src/tensor/tensor.h"

using namespace einsum::trees;

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
void test_model(uint32_t batch_size) {
    std::string str_repr = "[[[1,0],[2,1]->[2,0]r],[3,2]->[3,0]r],[4,3]->[4,0]";
    EinsumTree model_tree = EinsumTree(str_repr, {batch_size, 4, 64, 16, 3}, true);
    model_tree.optimize();
    model_tree.lower();

    std::cout << "Running model ..." << std::endl;
    std::cout << "*******************************************" << std::endl;
    std::cout << "String representation: " << str_repr << std::endl;
    std::cout << "Sizes: (" << batch_size << ", 4, 64, 16, 3)" << std::endl;
    std::cout << "*******************************************" << std::endl;
    model_tree.print();

    Tensor W1 = Tensor(64, 4);
    Tensor b1 = Tensor(64);
    Tensor W2 = Tensor(64, 16);
    Tensor b2 = Tensor(16);
    Tensor W3 = Tensor(16, 3);
    Tensor b3 = Tensor(3);

    // input
    Tensor input = Tensor(4, (int)batch_size);
    srand48(time(NULL));
    // init tensors with random values
    for (size_t i = 0; i < 4 * batch_size; i++) {
        input.data[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 4 * batch_size; i++) {
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

    float* model_output = new float[3 * batch_size];

    for (size_t i = 0; i < 3 * batch_size; i++) {
        model_output[i] = 0.0f;
    }

    // benchmark execution
    int64_t reps = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < reps; i++) {
        model_tree.execute(model_inputs, model_bias, model_output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "*******************************************" << std::endl;
    std::cout << "Model execution completed." << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Execution time (seconds): " << duration.count() / 1000.0 << " s" << std::endl;
    std::cout << "*******************************************" << std::endl;
}

int main(int argc, char* argv[]) {
    uint32_t batch_size = 1;

    if (argc > 1) {
        batch_size = std::stoi(argv[1]);
    } else {
        std::cout << "No batch size provided, using default: 1" << std::endl;
    }

    test_model(batch_size);

    return 0;
}