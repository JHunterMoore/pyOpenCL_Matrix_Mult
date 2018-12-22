# pyOpenCL_Matrix_Mult
Matrix Multiplication demonstration using pyOpenCL with a CSV implementation for added usability.  This is more an exploration/research project that started as school assignment.  The future of this project will be to continue finding more applications for pyOpenCL as a cross-platform heterogeneous parallelization environment.


# MatrixMultiplier.py
This is the CSV read-write implementation, meant to give this project a little more general usability. Currently runs by itself in a command-line and asks for each file name one-at-a-time.  I've attempted to keep it as general use as possible, so it may be used by any machine whether it has a GPU or only a CPU thanks to code from those credited in the source code.  

I've especially been impressed with pyOpenCL's ability to let this code be so portable that I can keep it on a flash drive and use it on machines that run AMD, NVidia, Intel or no GPU at all.  The only drawback so far has been some issues with machines that have CUDA installed, and making sure to have pybind11 and the proper OpenCL headers on the host machine.

# plainMul.py and OpenCLMatrixMultTest.py

These act as a test of pyOpenCL itself against a simple serial implementation.  Both include timers around the operation itself and statements to output the results.  To keep it simple.  I used a square matrix with a size that can easily be defined in the source code, and generated them using the same random seed.
