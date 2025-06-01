#ifndef TENGEN_TENSOR_H
#define TENGEN_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>

#include "Fastor/Fastor.h"

namespace TenGen {
    // e.g. Tensor<float, 3, 3, 4>
    template <typename T, size_t... Dims>
    class Tensor {
       public:
        //
        // --------------------- Define some tensor attributes ---------------------
        //

        // store the dtype of the tensor, e.g. Tensor<float, 2, 3, 4>::dtype // == float
        using dtype = T;

        // store the size of the whole tensor, e.g. Tensor<float, 2, 3, 4>::size // == 24
        static constexpr size_t size = (Dims * ...);

        // e.g. Tensor<float, 2, 3, 4>::Dims == {2, 3, 4}
        // storage for the elements
        static constexpr std::array<T, size> _data;

        // store the shape of the tensor
        static constexpr std::array<size_t, sizeof...(Dims)> shape = {Dims...};

        // store the rank of the tensor, e.g. Tensor<float, 2, 3, 4>::rank  // == 3
        static constexpr size_t rank = shape.size();

        // store the stride of each dimension
        // e.g. Tensor<float, 2, 3, 4>::strides == {12, 4, 1}
        // S2 = 1                   last dimension stride is always 1
        // S1 = D2 * S2 = 4 * 1 = 4
        // S0 = D1 * S1 = 3 * 4 = 12
        static constexpr std::array<size_t, rank> strides = [] {
            std::array<size_t, rank> s = {};
            size_t stride = 1;
            for (size_t i = rank; i-- > 0;) {
                s[i] = stride;
                stride *= shape[i];
            }
            return s;
        }();

        //
        // ------------------------------ Overloads ---------------------------------
        //

        // overload of the subscript operator and
        // returns a reference to the underlying data element at that index (flat)
        T& operator[](size_t index) {
            return _data[index];
        }

        //
        // ------------------------------ Functions ---------------------------------
        //

        // multidimensional access using indices
        template <typename... Indices>
        auto at(Indices... indices) {
            // store the indices in an array
            std::array<size_t, rank> idx = {static_cast<size_t>(indices)...};
            size_t flat_index = 0;

            // for each index, calculate the flat index
            // e.g. index = i * S0 + j * S1 + k * S2 + ... + n * Sn
            for (size_t i = 0; i < rank; ++i) {
                flat_index += idx[i] * strides[i];
            }

            return _data[flat_index];
        }

        // fills the tensor with random values (integers or floats)
        void fillRandom() {
            for (size_t i = 0; i < size; ++i) {
                if constexpr (std::is_floating_point<dtype>::value) {
                    _data[i] = static_cast<dtype>(rand()) / static_cast<dtype>(RAND_MAX);  // e.g. 0.0 - 1.0

                } else {
                    _data[i] = static_cast<dtype>(rand() % 100);  // e.g. integer in [0, 99]
                }
            }
        }

        //
        // --------------------------- Export / Import ------------------------------
        //

        // returns the raw data as a real copy
        std::array<dtype, size> exportData() {
            return _data;
        }

        // exports the pointer to the underlying data
        // Note: this is a pointer to the data, so manipulating the data
        // will also manipulate the data in the tensor
        dtype* exportPointer() {
            return _data.data();
        }

        // create a real copy from the fastor tensor
        // Note: this will copy the data from the fastor tensor to this tensor
        // and therefore the fastor tensor must have the same shape and size
        // and its overriding the data of this tensor
        void importData(const std::array<dtype, size>& new_data) {
            _data = new_data;
        }

        void fromFastor(const Fastor::Tensor<T, Dims...>& fastor_tensor) {
            std::copy_n(fastor_tensor.data(), _data.size(), _data.data());
        }

        //
        // -------------------------------- DEBUG -----------------------------------
        //

        // DEBUG function to print the infos of the tensor
        void info() const {
            std::cout << "Tensor Info:" << std::endl;
            std::cout << "  Rank: " << rank << std::endl;
            std::cout << "  Shape: ";
            for (const auto& dim : shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  Strides: ";
            for (const auto& stride : strides) {
                std::cout << stride << " ";
            }
            std::cout << std::endl;
            std::cout << "  Size: " << size << std::endl;
        }
        // DEBUG function to print the tensor data
        void print_raw() const {
            std::cout << "Tensor Data (raw):" << std::endl;
            for (size_t i = 0; i < size; ++i) {
                std::cout << _data[i] << " ";
            }
            std::cout << std::endl;
        }

        // // DEBUG: view of the tensor as a 1D array
        // template <typename... Indices>
        // auto view1D(Indices... indices) {

        // }

        // // DEBUG: view of the tensor as a 2D array
        // template <typename... Indices>
        // auto view2D(Indices... indices) {
        //     ...  // implementation for 2D view
        // }
    };
}  // namespace TenGen

#endif  // TENSOR_TENSOR_TENSOR_H