#include <stdio.h>
#include "cublas_v2.h"

#define n 1000000
int main(void) {
    
    // Create a Host vector
    float* x;
    x = (float*)malloc(n * sizeof(*x));

    for(int j=0;j<n;j++)
        x[j] = (float)j;

    // Create a Device vector
    float* d_x;
    cudaMalloc((void**)&d_x,n*sizeof(*x));

    // Create a cuBLAS
    cublasHandle_t handle;

    cublasCreate(&handle);
    cublasSetVector(n,sizeof(*x),x,1,d_x,1);

    int result;
    cublasIsamax(handle, n, d_x, 1, &result);
    printf("Max: %d \n", result - 1);

    cublasIsamin(handle, n, d_x, 1, &result);
    printf("Min: %d \n", result);

    // Clean up the program
    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    
    return 0;
}