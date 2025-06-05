#ifndef TENGEN_STRUCTS_H
#define TENGEN_STRUCTS_H

#include <cstdint>
#include <span>
#include <vector>

#include "TenGen/types/Types.h"

using namespace TenGen::Types;

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

#endif  // TENGEN_STRUCTS_H
