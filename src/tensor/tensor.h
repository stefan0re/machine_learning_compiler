// tensor.h

#ifndef TENSOR_H
#define TENSOR_H

#include <sstream>
#include <vector>

class Tensor {
   public:
    struct DimInfo {
        int dim_t = 99;
        int dim_sizes;
        int stride;
        int loop_id;
        int exec_t = 99;
    };

    std::vector<DimInfo> id;

    // constructor taking n dimension sizes
    // template definitions must be visible to every translation unit that uses them,
    // so define it in the header
    template <typename... Dims>
    Tensor(Dims... dims) {
        // store each dimension size in a vector
        std::vector<int> sizes = {dims...};

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

    void swap(int i, int j);
    void info() const;
    std::string info_str() const;
};

#endif  // TENSOR_H
