import torch
import sys
import time


M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])

A = torch.randn(M, K, dtype=torch.float32)
B = torch.randn(K, N, dtype=torch.float32)
C = torch.zeros(M, N, dtype=torch.float32)

# benchmark GFLOPS
start = time.perf_counter()
for _ in range(100):
    C = torch.mm(A, B)
end = time.perf_counter()  
exec_time = end - start
print(f"Model evaluation took {exec_time:.8f} seconds")
flops = 2 * M * N * K * 100

print(f"GFLOPs: {(flops / exec_time) * 1e-9:,}")