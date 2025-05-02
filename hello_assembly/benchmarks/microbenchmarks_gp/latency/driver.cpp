#include <cstdint>
#include <chrono>
#include <iostream>

extern "C" {
  uint64_t add_latency( uint64_t repetitions );
  uint64_t add_throughput( uint64_t repetitions );
  uint64_t mul_latency( uint64_t repetitions );
  uint64_t mul_throughput( uint64_t repetitions );
}

int main() {

  
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration< double > duration;
  uint64_t reps = 5000000;
  double gflops = 0;

  /**
    * latency test for add instruction
    */

  start = std::chrono::high_resolution_clock::now();
  gflops = add_latency( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Latency Test ADD..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GOPS: "    << gflops               << std::endl;


  reps *= 1000;
  start = std::chrono::high_resolution_clock::now();
  gflops = add_throughput( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Throughput Test ADD..."        << std::endl;
  std::cout << "  Duration: " << duration.count()  << std::endl;
  std::cout << "  GOPS: " << gflops                << std::endl;

  /**
    * latency test for mul instruction
    */
  
  reps /= 4000;

  start = std::chrono::high_resolution_clock::now();
  gflops = mul_latency( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Latency Test MUL..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GOPS: " << gflops               << std::endl;

  reps *= 1000;

  start = std::chrono::high_resolution_clock::now();
  gflops = mul_throughput( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Throughput Test MUL..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GOPS: " << gflops               << std::endl;
  std::cout << "---------------------------------" << std::endl;




  return EXIT_SUCCESS;
}