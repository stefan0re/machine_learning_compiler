#include <chrono>
#include <iostream>
#include <string>

#include "../src/einsum/trees/einsum_trees.h"
#include "../src/mini_jit/include/gemm_ref.h"

using namespace einsum::trees;

/**
 * Contraction String: [[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]
 * Dimension sizes: 0: 100
 *                  1: 72,
 *                  2: 128,
 *                  3: 128,
 *                  4: 3,
 *                  5: 71,
 *                  6: 305,
 *                  7: 32,
 *                  8: 3
 * Notation; Node ID / First Touch | Operation Primitive | Last Touch
    └─ 0,1,2,3,4; 0 / 99 | 3 | 99
       ├─ 7,3,4; 1 / 99 | 3 | 99
       │  ├─ 8,4; 2 / 99 | 99 | 99
       │  └─ 7,3,8; 3 / 99 | 99 | 99
       └─ 0,1,2,7; 4 / 99 | 3 | 99
           ├─ 1,2,5,7; 5 / 99 | 3 | 99
           │  ├─ 2,6,7; 6 / 99 | 99 | 99
           │  └─ 1,5,6; 7 / 99 | 99 | 99
           └─ 0,5; 8 / 99 | 99 | 99
    Leaf ID's: 2 3 6 7 8
 */
void first_example() {
    std::cout << "Running first pbtc example..." << std::endl;
    std::string str_repr = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
        tree.lower();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Compiling first example: " << elapsed.count() / 50 << " seconds" << std::endl;

    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.lower();

    float* in0 = new float[3 * 3];
    float* in1 = new float[32 * 128 * 3];
    float* in2 = new float[128 * 305 * 32];
    float* in3 = new float[72 * 71 * 305];
    float* in4 = new float[100 * 71];
    float* out = new float[100 * 72 * 128 * 128 * 3];

    srand48(time(NULL));
    for (size_t i = 0; i < 3 * 3; i++) {
        in0[i] = (float)drand48();
    }

    for (size_t i = 0; i < 32 * 128 * 3; i++) {
        in1[i] = (float)drand48();
    }

    for (size_t i = 0; i < 128 * 305 * 32; i++) {
        in2[i] = (float)drand48();
    }

    for (size_t i = 0; i < 72 * 71 * 305; i++) {
        in3[i] = (float)drand48();
    }

    for (size_t i = 0; i < 100 * 71; i++) {
        in4[i] = (float)drand48();
    }

    for (size_t i = 0; i < 100 * 72 * 128 * 128 * 3; i++) {
        out[i] = 0.0f;
    }

    start = std::chrono::high_resolution_clock::now();
    tree.execute({in0, in1, in2, in3, in4}, {}, out);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "  Execution first example: " << elapsed.count() << " seconds" << std::endl;
    double ops = tree.operations();
    std::cout << "  Operations in first example: " << ops << std::endl;
    double gflops = ops / elapsed.count() / 1e9;
    std::cout << "  GFLOPS for first example: " << gflops << std::endl;

    delete[] in0;
    delete[] in1;
    delete[] in2;
    delete[] in3;
    delete[] in4;
    delete[] out;
    tree.delete_tree();
    std::cout << "Finished first pbtc example..." << std::endl;
}

/**
 * Contraction String: [[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]
 * Dimension sizes: 0: 60
 *                  1: 60,
 *                  2: 20,
 *                  3: 20,
 *                  4: 8,
 *                  5: 8,
 *                  6: 8,
 *                  7: 8,
 *                  8: 8,
 *                  9: 8
 * Notation; Node ID / First Touch | Operation Primitive | Last Touch
    └─ 0,1,2,3; 0 / 99 | 3 | 99
       ├─ 0,4,7,8,2,3; 1 / 99 | 3 | 99
       │  ├─ 7,8,5,6,2,3; 2 / 99 | 3 | 99
       │  │  ├─ 8,6,9,3; 3 / 99 | 1 | 99
       │  │  │  └─ 3,6,8,9; 4 / 99 | 99 | 99
       │  │  └─ 7,5,2,9; 5 / 99 | 1 | 99
       │  │     └─ 2,5,7,9; 6 / 99 | 99 | 99
       │  └─ 0,4,5,6; 7 / 99 | 99 | 99
       └─ 1,4,7,8; 8 / 99 | 99 | 99
    Leaf ID's: 4 6 7 8
    Bias ID's: 0 1 2
 *
 */

void second_example() {
    std::cout << "Running second pbtc example..." << std::endl;
    std::string str_repr = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";

    auto start = std::chrono::high_resolution_clock::now();
    /*for (int i = 0; i < 50; i++) {
        EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
        tree.lower();
    }*/
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Compiling second example: " << elapsed.count() / 50 << " seconds" << std::endl;

    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.lower();
    tree.print();

    float* in0 = new float[20 * 8 * 8 * 8];
    float* in1 = new float[20 * 8 * 8 * 8];
    float* in2 = new float[60 * 8 * 8 * 8];
    float* in3 = new float[60 * 8 * 8 * 8];
    float* out = new float[60 * 60 * 20 * 20];

    srand48(time(NULL));
    for (size_t i = 0; i < 20 * 8 * 8 * 8; i++) {
        in0[i] = (float)drand48();
    }

    for (size_t i = 0; i < 20 * 8 * 8 * 8; i++) {
        in1[i] = (float)drand48();
    }

    for (size_t i = 0; i < 60 * 8 * 8 * 8; i++) {
        in2[i] = (float)drand48();
    }

    for (size_t i = 0; i < 60 * 8 * 8 * 8; i++) {
        in3[i] = (float)drand48();
    }

    for (size_t i = 0; i < 60 * 60 * 20 * 20; i++) {
        out[i] = 0.0f;
    }

    start = std::chrono::high_resolution_clock::now();
    tree.execute({in0, in1, in2, in3}, {}, out);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "  Execution second example: " << elapsed.count() << " seconds" << std::endl;
    double ops = tree.operations();
    std::cout << "  Operations in second example: " << ops << std::endl;
    double gflops = ops / elapsed.count() / 1e9;
    std::cout << "  GFLOPS for second example: " << gflops << std::endl;

    delete[] in0;
    delete[] in1;
    delete[] in2;
    delete[] in3;
    delete[] out;
    tree.delete_tree();
    std::cout << "Finished second pbtc example..." << std::endl;
}

int main() {
    std::cout << "Benchmarking Einsum Strings..." << std::endl;
    first_example();
    second_example();
    return EXIT_SUCCESS;
}