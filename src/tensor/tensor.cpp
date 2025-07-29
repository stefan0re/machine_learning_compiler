#include "tensor.h"

#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// constuctor that gets the dimension sizes as vector
Tensor::Tensor(std::vector<u_int32_t> dims) {
    std::vector<int> sizes(dims.begin(), dims.end());
    setup(sizes);

    size = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    data = new float[size];
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

// compare a tensor with another
bool Tensor::compare(Tensor& tensor, float delta) {
    bool ret = true;

    // check for equal size first
    if (size == tensor.size) {
        for (int i = 0; i < size; i++) {
            // if the error is bigger than the delta
            if (delta < abs(data[i] - tensor.data[i])) {
                std::cout << "Differe in " << i << ": " << data[i] << " != " << tensor.data[i] << std::endl;
                ret = false;
            }
        }

        return ret;

    } else {
        std::cout << "Tensors have diffrent sizes." << std::endl;
        return false;
    }
}

// print 1D Tensors
void Tensor::print() {
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " , ";
    }
    std::cout << std::endl;
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

std::vector<Tensor> Tensor::from_torchpp(std::string path, int in_size) {
    std::ifstream file(path);
    std::string line;
    std::vector<Tensor> layers;

    int out_size;
    int line_count = 0;

    // for each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;

        // parse floats from line
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stof(token));
        }

        // if the line is a layer (matrix)
        if (line_count % 2 == 0) {
            // calculate the size of the output layer
            out_size = values.size() / in_size;
            // create the tensor
            // this is the pytorch row-major standart
            Tensor t = Tensor(out_size, in_size);
            // update the in_size to the ouput layer so it
            // becomes the new input layer
            in_size = out_size;

            // allocate raw float array and copy values
            t.data = new float[values.size()];
            for (size_t i = 0; i < values.size(); ++i) {
                t.data[i] = values[i];
            }

            // store the tensor
            layers.push_back(t);

        }
        // otherwise it must be a baies (vector)
        else {
            // create a vector
            Tensor t = Tensor(static_cast<int>(values.size()));

            // allocate raw float array and copy values
            t.data = new float[values.size()];
            for (size_t i = 0; i < values.size(); ++i) {
                t.data[i] = values[i];
            }

            // store the tensor
            layers.push_back(t);
        }
        std::cout << line_count << std::endl;
        line_count++;
    }

    return layers;
}

Tensor Tensor::from_csv(std::string path) {
    std::ifstream file(path);
    std::string line;
    std::vector<float> float_values;
    int line_count = 0;
    int element_count = 0;

    // for each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        element_count = 0;

        while (std::getline(ss, token, ',')) {
            try {
                // try to convert to float
                float f = std::stof(token);
                float_values.push_back(f);
                element_count++;
            } catch (const std::invalid_argument&) {
                // Not a valid float, treat as string
                // stringValues.push_back(token);
            }
        }

        line_count++;
    }

    Tensor t = Tensor(line_count, element_count);

    t.data = new float[float_values.size()];
    for (size_t i = 0; i < float_values.size(); ++i) {
        t.data[i] = float_values[i];
    }

    return t;
}