#include "einsum_ref.h"

std::vector<int64_t> prime_factors(int64_t n) {
    std::vector<int64_t> factors;

    if (n == 0 || n == 1 || n == -1)
        return factors;

    if (n < 0) {
        factors.push_back(-1);
        n = -n;
    }

    while ((n & 1) == 0) {
        factors.push_back(2);
        n >>= 1;
    }

    for (int64_t p = 3; p * p <= n; p += 2) {
        while (n % p == 0) {
            factors.push_back(p);
            n /= p;
        }
    }

    if (n > 1)
        factors.push_back(n);

    return factors;
}

int64_t find_new_size(std::vector<int64_t> const& i_sizes) {
    int64_t l_new_size = 1;
    for (size_t i = 0; i < i_sizes.size(); i++) {
        l_new_size *= i_sizes[i];
        if (l_new_size > 64) {
            l_new_size /= i_sizes[i];
            break;
        }
    }

    if (l_new_size == 1) {
        l_new_size = i_sizes[0];
    }

    return l_new_size;
}