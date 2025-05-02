#include <cstdint>
#include <iostream>

extern "C" void add_values(int32_t* a, int32_t* b, int32_t* c);

int main() {
    int32_t a, b, c;
    a = 32;
    b = 10;
    add_values(&a, &b, &c);

    std::cout << a << " + " << b << " = " << c << std::endl;
    return 0;
}