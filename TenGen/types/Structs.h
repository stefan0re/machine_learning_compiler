#ifndef TENGEN_Structs_H
#define TENGEN_Structs_H

#include <cstdint>

namespace TenGen::Structs {
    struct KernelSize {
        int M;
        int N;
    };

    struct KernelSizes {
        KernelSize kernel1;
        KernelSize kernel2;
        KernelSize kernel3;
        KernelSize kernel4;
    };

    struct AreaDefinition {
        int32_t m;
        int32_t n;
        int32_t m_iters;
        int32_t n_iters;
        uint32_t offset;
        KernelSize kernelsize;
    };

}  // namespace TenGen::Structs

#endif  // TENGEN_Structs_H
