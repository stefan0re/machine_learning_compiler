#include <iostream>

#include "../src/tensor/tensor.h"

int main(int argc, char const *argv[]) {
    Tensor tens = Tensor::from_csv("iris.csv");
    std::cout << "\nTensor with: " << std::endl;
    tens.info();
    return 0;
}
