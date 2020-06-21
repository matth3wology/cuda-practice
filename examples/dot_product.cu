#include <stdio.h>
#include "cublas_v2.h"

#define N 100

int main() {

    // X Vector
    float* d_x;
    cudaMallocManaged(&d_x, N * sizeof(float));
    for(int i=0;i<N;i++)
        d_x[i] = (float)i;

    // Y Vector
    float* d_y;
    cudaMallocManaged(&d_y, N * sizeof(float));
    for(int i=0;i<N;i++)
        d_y[i] = (float)i;

    //Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float result;
    // y = dot(x,y)
    cublasSdot(handle, N, d_x, 1, d_y, 1, &result);
    cudaDeviceSynchronize();

    // Print y
    printf("Dot Product: %f \n", result);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);

    return 0;
}