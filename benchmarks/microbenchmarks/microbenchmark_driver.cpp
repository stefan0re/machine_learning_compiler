#include <chrono>
#include <cstdint>
#include <iostream>

extern "C" void benchmark_add_shifted_registers(int32_t iterations);
extern "C" void benchmark_mul(int32_t iterations);

int main() {
    double duration, throughput;
    std::chrono::_V2::system_clock::time_point start, end;

    int32_t iterations = 1000000000;

    // add shifted registers
    start = std::chrono::high_resolution_clock::now();
    benchmark_add_shifted_registers(iterations);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - start).count();
    throughput = (iterations / duration) * 8;  // 8 ops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "ADD shifted registers" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GOPS\n"
              << std::endl;

    // mul
    start = std::chrono::high_resolution_clock::now();
    benchmark_mul(iterations);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - start).count();
    throughput = (iterations / duration) * 8;  // 8 ops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "MUL" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GOPS\n"
              << std::endl;

}