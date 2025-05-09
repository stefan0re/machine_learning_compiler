import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('data/bm_16x6xK.csv', delimiter=',')



x1, y1 = data1[:,2], data1[:,4]



plt.plot(x1,y1)


plt.grid(True)

plt.xticks(np.arange(0, 129, 16))
plt.xlabel('K ')
plt.ylabel('GFLOPS')

plt.legend(title='GEMM M = 16, N = 6')

plt.show()
