#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../src/tensor/tensor.h"

// ReLU activation
void relu(Tensor& vec) {
    for (size_t i = 0; i < vec.size; ++i) {
        vec.data[i] = std::max(0.0f, vec.data[i]);
    }
}

// Matrix-vector multiplication: y = x * W^T + b
// in row-major access pattern
void matmul_add(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& out) {
    /*
    W = [w1.1, w1.2, w1.3, w1.4,        [x1, x5         [y1, y_n
         w2.1, w2.2, w2.3, w2.4,   *     x2, x6 ...  =   y2, y_n+1
         w3.1, w3.2, w3.3, w3.4,         x3, x7          y3, y_n+2
                ......          ]        x4, x8    ]        ...    ]

    -> y1 = w1.1 * x1 + w1.2 + w1.3 * w3 + w1.4 * w4 + b1.1
    -> Tensor(64, 4) -> stride: (4, 1)
    */

    size_t batch_size = x.id[0].dim_sizes;
    size_t input_dim = x.id[1].dim_sizes;
    size_t output_dim = W.id[0].dim_sizes;

    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t o = 0; o < output_dim; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < input_dim; ++i) {
                size_t x_index = b_idx * x.id[0].stride + i * x.id[1].stride;
                size_t W_index = o * W.id[0].stride + i * W.id[1].stride;
                sum += x.data[x_index] * W.data[W_index];
            }
            size_t out_index = b_idx * out.id[0].stride + o * out.id[1].stride;
            out.data[out_index] = sum + b.data[o];
        }
    }
}

// Forward pass of BasicNet
void forward(const Tensor& input,                 // size = b = 4
             const Tensor& W1, const Tensor& b1,  // W1: [c][b], b1: [c]
             const Tensor& W2, const Tensor& b2,  // W2: [d][c], b2: [d]
             const Tensor& W3, const Tensor& b3,  // W3: [e][d], b3: [e]
             Tensor& output)                      // output: size = e
{
    // Temporary tensors
    Tensor z1 = Tensor(4, 64);
    Tensor z2 = Tensor(4, 16);

    // fc1
    matmul_add(input, W1, b1, z1);
    relu(z1);

    // fc2
    matmul_add(z1, W2, b2, z2);
    relu(z2);

    // fc3
    matmul_add(z2, W3, b3, output);

    delete[] z1.data;
    delete[] z2.data;
}

TEST_CASE("Model::BasicNet::Output", "[Model][BasicNet][Output]") {
    // load the trained model
    std::vector<Tensor> model = Tensor::from_torchpp("../../python/data/model.torchpp", 4);

    Tensor W1 = model[0];
    Tensor b1 = model[1];
    Tensor W2 = model[2];
    Tensor b2 = model[3];
    Tensor W3 = model[4];
    Tensor b3 = model[5];

    // load example input and output
    std::vector<Tensor> example = Tensor::load_example("../../python/data/example.csv", 4);

    // convert the vetor to a tensor
    Tensor input = example[0];

    // output
    Tensor output = Tensor(4, 3);
    Tensor output_ref = example[1];

    // DEBUG
    // input.print();
    // output_ref.print();
    // b1.print();

    forward(input, W1, b1, W2, b2, W3, b3, output);

    REQUIRE(output.compare(output_ref, 0.0001));
}
