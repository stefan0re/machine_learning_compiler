#ifndef TENGEN_TENSOR_H
#define TENGEN_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>

namespace TenGen {
    // e.g. Tensor<float, 3, 3, 4>
    template <typename T, size_t... Dims>
    class Tensor {
        // e.g. == static constexpr size_t NumElements = 3 * 3 * 4;
        static constexpr size_t NumElements = (Dims * ...);
        std::array<T, NumElements> data_;

       public:
        using value_type = T;
        // e.g. Tensor<float, 2, 3, 4>::rank  // == 3
        static constexpr size_t rank = sizeof...(Dims);

        // generate the default constructor
        Tensor() = default;

        // overload of the subscript operator and
        // returns a reference to the underlying data element at that index (flat)
        T& operator[](size_t index) {
            return data_[index];
        }

        // get total size
        constexpr size_t size() const { return NumElements; }

        // multidimensional access (only for 3D in this example)
        template <typename... Indices>
        T& at(Indices... indices) {
            return data_[flatten(indices...)];
        }

       private:
        // flatten multidim index to linear offset used for access in multidim noation
        // flat_index = i * D1 * D2 + j * D3 + k
        // e.g. flatten(1, 2, 3)
        template <typename... Indices>
        constexpr size_t flatten(Indices... indices) const {
            // shape[] is array: {3, 3, 4}
            constexpr size_t shape[] = {Dims...};
            // captures the passed indice: e.g. {i, j, k}
            size_t idx[] = {static_cast<size_t>(indices)...};
            size_t offset = 0;
            for (size_t i = 0; i < sizeof...(Dims); ++i) {
                offset = offset * shape[i] + idx[i];
            }
            return offset;
        }
    };
}  // namespace TenGen

#endif  // TENSOR_TENSOR_TENSOR_H