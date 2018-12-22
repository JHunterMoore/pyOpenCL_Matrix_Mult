import numpy as np
import time

MATRIX_SIZE = 200
DIMENSIONS = np.array([MATRIX_SIZE,MATRIX_SIZE], dtype=np.int64)

np.random.seed(9353)

m1 = np.array(np.random.randint(10, size=DIMENSIONS),dtype=np.int32)
m2 = np.array(np.random.randint(10, size=DIMENSIONS),dtype=np.int32)
result  = np.zeros_like(m1)

print("Matrices Initialized")

start_time = time.time()

for i in range(MATRIX_SIZE):
    for j in range(MATRIX_SIZE):
        for k in range(MATRIX_SIZE):
            result[i][j] += m1[i][k] * m2[k][j]

finish_time = time.time() - start_time
print("Time to finish: " + str(finish_time))

print(m1)
print(m2)
print(result)        #show result_g result hasn't been passed the finished matrix

