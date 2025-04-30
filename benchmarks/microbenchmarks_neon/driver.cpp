#include <cstdint>
#include <chrono>
#include <iostream>

extern "C" {
  uint64_t fmla_throughput_4s( uint64_t repetitions );
  uint64_t fmla_throughput_2s( uint64_t repetitions );
  uint64_t fmadd_throughput( uint64_t repetitions );
  uint64_t fmla_latency_src( uint64_t repetitions );
  uint64_t fmla_latency_dst( uint64_t repetitions );

}

void throughput_bench() {
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration< double > duration;
  uint64_t reps = 15000000;
  double gflops = 0;

  /**
    * throughput fmla
    */
  // fmla 4s
  start = std::chrono::high_resolution_clock::now();
  gflops = fmla_throughput_4s( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Throughput FMLA 4S..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GFLOPS: "    << gflops               << std::endl;


  // fmla 2s
  start = std::chrono::high_resolution_clock::now();
  gflops = fmla_throughput_2s( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Throughput FMLA 2S..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GFLOPS: "    << gflops               << std::endl;

  // fmadd scalar
  start = std::chrono::high_resolution_clock::now();
  gflops = fmadd_throughput( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Throughput FMADD..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GFLOPS: "    << gflops               << std::endl;

  
}

void latency_bench(){
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration< double > duration;
  uint64_t reps = 1000000;
  double gflops = 0;

  /**
    * latency fmla src reg
    */
  start = std::chrono::high_resolution_clock::now();
  gflops = fmla_latency_src( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Latency SRC FMLA 4S..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GFLOPS: "    << gflops               << std::endl;


  /**
    * latency fmla dst reg
    */
  start = std::chrono::high_resolution_clock::now();
  gflops = fmla_latency_dst( reps );
  end   = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - start);

  gflops *= reps;
  gflops *= 1.0E-9;
  gflops /= duration.count();



  std::cout << "---------------------------------" << std::endl;
  std::cout << "Latency DST FMLA 4S..."           << std::endl;
  std::cout << "  Duration: " << duration.count()    << std::endl;
  std::cout << "  GFLOPS: "    << gflops               << std::endl;
}

int main() {

  throughput_bench();
  latency_bench();

  return EXIT_SUCCESS;
}