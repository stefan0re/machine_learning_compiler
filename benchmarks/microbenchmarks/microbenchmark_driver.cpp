#include <chrono>
#include <cstdint>
#include <iostream>

extern "C" void benchmark_add_shifted_registers(int32_t iterations);

int main() {
    double duration, throughput;
    std::chrono::_V2::system_clock::time_point start, end;

    int iterations = 1000000000;

    // add shifted registers
    start = std::chrono::high_resolution_clock::now();
    benchmark_add_shifted_registers(iterations);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - start).count();
    throughput = (iterations * 8) / duration;  // 8 ops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "ADD shifted registers" << std::endl;
    std::cout << "Duration:\t" << duration << "sec" << std::endl;
    std::cout << "Throughput:\t" << throughput << "ops/sec\n"
              << std::endl;
}