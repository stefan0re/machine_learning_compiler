#include "tensor.h"

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <vector>

// swap two DimInfo's
void Tensor::swap(int i, int j) {
    // "Out of Bounds"-check
    if (i < 0 || j < 0 || i >= static_cast<int>(id.size()) || j >= static_cast<int>(id.size())) {
        throw std::out_of_range("Swap indices out of range.");
    }
    std::swap(id[i], id[j]);
}

// utility to print the Tensor structure
void Tensor::info() const {
    for (const auto& dim : id) {
        std::cout << "DimInfo { dim_t: " << dim.dim_t
                  << ", dim_sizes: " << dim.dim_sizes
                  << ", stride: " << dim.stride
                  << ", loop_id: " << dim.loop_id
                  << ", exec_t: " << dim.exec_t
                  << " }\n";
    }
}

// return the info as a string (mostly for catch2)
std::string Tensor::info_str() const {
    std::ostringstream oss;
    for (const auto& dim : id) {
        oss << "DimInfo { dim_t: " << dim.dim_t
            << ", dim_sizes: " << dim.dim_sizes
            << ", stride: " << dim.stride
            << ", loop_id: " << dim.loop_id
            << ", exec_t: " << dim.exec_t
            << " }\n";
    }
    return oss.str();
}