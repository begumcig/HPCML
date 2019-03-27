import numpy as np
import sys
import time


def dp(N,A,B):
	R = 0.0;
	for j in range(0,N):
		R += A[j]*B[j]
	return R

#### We want exactly 2 arguments, array size and loop count. ####
if len(sys.argv) != 3:
	print("Too many or too little arguments!")
	exit()


print ("\n---------- NOTE ---------- \nPrinting the result at every loop to avoid optimization.\nPlease do not remove.")
print("--------------------------\n")

####Initializations####
N = int(sys.argv[1])
iterc = int(sys.argv[2])

A = np.ones(N,dtype=np.float32)
B = np.ones(N,dtype=np.float32)
timeElapsArr = np.zeros(iterc, dtype =np.float32)

####Benchmark Calculation####
for i in range (0, iterc):
	start = time.monotonic()
	res = dp(N, A, B)
	print (f'result is: {res}')
	end  = time.monotonic()
	timeElapsArr[i] = end-start
	print(f'R: {res}   T: {end-start}')


####Time Average####
timeSum = sum(timeElapsArr[int(iterc/2) : iterc])
timeAvg = timeSum / (iterc/2)

####Performance####
flops = (N * 2) / (timeAvg * 1000000000)
gbs = (N * 2 * np.dtype(np.float32).itemsize) / (timeAvg * 1000000000)
print (f"Avg T: {timeAvg} sec  B: {gbs} GB/sec   F: {flops } GFLOP/sec")
