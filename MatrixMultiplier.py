# OpenCL Matrix Multiplication w/ CSV I/O
# By: J. Hunter Moore
# Date: 21 November 2018
#
# This program is a use application of the OpenCL Matrix Multiplication program
# made earlier that takes csv input of 2 matrices, calculates the product, and
# then outputs the resulting matrix to a file the user chooses
#
# Credits to: PyOpenCL documentation (https://documen.tician.de/pyopencl)
#             inducer on github (https://github.com/inducer/pyopencl)
#             scipy's numpy reference (https://docs.scipy.org/doc/numpy/reference/)
#             AndreasKloeckner on wiki.tiker.net (https://wiki.tiker.net/PyOpenCL)
#             Python's csv documentation (https://docs.python.org/3/library/csv.html)

import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
import pyopencl as cl
import numpy as np
import time
import csv

m1File = input("Enter file for first matrix: ")
m1 = np.loadtxt(m1File, dtype=np.int32, delimiter=",")

m2File = input("Enter file for second matrix: ")
m2 = np.loadtxt(m2File, dtype=np.int32, delimiter=",")

if m1.shape[1] == m2.shape[0]:

#This defines the global work group size  Your device may either use a 64-bit
#or 32-bit.  If result values are strange, try switching to 32-bit
    DIMENSIONS = np.array([m1.shape[0],m2.shape[1]],dtype=np.int64)
    #DIMENSIONS = np.array([m1.shape[0],m2.shape[1]],dtype=np.int32)

    result = np.full(DIMENSIONS, 0, dtype=np.int32)

    print("Matrices Initialized")

# Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    CPU = plat[0].get_devices()
    try:
        GPU = plat[1].get_devices()
    except IndexError:
        GPU = "none"

#Create context for GPU/CPU (largely automated by pyopenCL)
    if GPU!= "none":
        ctx = cl.Context(GPU)
    else:
        ctx = cl.Context(CPU)

# Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

# get memory flags
    mf = cl.mem_flags

    print("Context and Queue initialized")

#Copy everything into the device
      #Since pyopencl has nice methods for array handling, we will use them on our matrices
    m1_g = cl_array.to_device(queue, m1)
    m2_g = cl_array.to_device(queue, m2)
    result_g = cl_array.to_device(queue, result)
      #for a single integer, we will need to load into the Buffer with the proper flags and data type
    width1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(m1.shape[1]))
    width2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(m2.shape[1]))

    print("Matrices moved to GPU")

#This is the C code for the kernel, compiled at runtime by openCL
    src = '''
    __kernel void matrixMult(__global int *m1, __global int *m2, __global int *result, __global int *width1, __global int *width2)
    {
        int w1 = *width1;
        int w2 = *width2;
        int posx = get_global_id(1);
        int posy = get_global_id(0);

        for(int i = 0; i < w1; i++)
           result[posy*w2+posx] += m1[posy*w1+i]*m2[i*w2+posx];
    }'''

#build the code from src
    prg = cl.Program(ctx, src).build()

    print("Program created.  Getting ready to start multiplying.")

#Mark start time
    start_time = time.time()
    ev = prg.matrixMult(queue, DIMENSIONS, None, m1_g.data, m2_g.data, result_g.data, width1_g, width2_g)

#Hold main until queue has emptied
    queue.finish()

#get total time
    finish_time = time.time() - start_time

    print("Finished Multiplying in: " + str(finish_time))

#If matrices are of reasonable size, display
    print(m1)
    print(m2)
    print(result_g)        #show result_g result hasn't been passed the finished matrix

    outfile = input("Where would you like to save this output: ")
    with open(outfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(result_g)
else:
    print("Matrices incompatible.")
