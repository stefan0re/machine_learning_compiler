#include "tensor.h"

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <vector>

// constuctor that gets the dimension sizes as vector
Tensor::Tensor(std::vector<u_int32_t> dims) {
    std::vector<int> sizes(dims.begin(), dims.end());
    setup(sizes);
}

// fill DimInfos and strides with the help of dimension size vector
void Tensor::setup(std::vector<int> sizes) {
    // of each dimension create a DimInfo
    for (int i = 0; i < sizes.size(); ++i) {
        DimInfo info;
        info.dim_sizes = sizes[i];
        info.loop_id = i;
        id.push_back(info);
    }

    // compute strides in reverse order
    // e.g. Tensor(2, 3, 4)::strides == {12, 4, 1}
    // S2 = 1                   last dimension stride is always 1
    // S1 = D2 * S2 = 4 * 1 = 4
    // S0 = D1 * S1 = 3 * 4 = 12
    int64_t stride = 1;
    for (size_t i = sizes.size(); i-- > 0;) {
        // assing the stride
        id[i].stride = stride;
        // multiple the current stride with the new one
        stride *= sizes[i];
    }
}

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