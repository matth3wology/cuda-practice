#include <stdio.h>
#include "cublas_v2.h"

#define N 100000

int main() {
    // Alpha Scalar
    float al = 2.0;

    // X Vector
    float* d_x;
    cudaMallocManaged(&d_x, N * sizeof(float));
    for(int i=0;i<N;i++)
        d_x[i] = (float)i;

    // Y Vector
    float* d_y;
    cudaMallocManaged(&d_y, N * sizeof(float));
    for(int i=0;i<N;i++)
        d_y[i] = -15.0;
        
    //Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // y = alpha * x + y
    cublasSaxpy(handle, N, &al, d_x, 1, d_y, 1);
    cudaDeviceSynchronize();

    // Print y
    printf("Y: ");
    for(int i=0;i<N;i++)
        printf(" %0.4f ", d_y[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_x);
    cublasDestroy(handle);

    return 0;
}