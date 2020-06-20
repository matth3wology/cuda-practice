#include <stdio.h>
#include "cublas_v2.h"

#define n 6
int main(void) {

    
    // Create a Vector and send to the Device
    float* x;
    cudaMallocManaged(&x, n*sizeof(*x));
    
    for(int j=0;j<n;j++)
    x[j] = (float)j;
    
    
    // Manage cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    int result;
    cublasIsamax(handle, n, x, 1, &result);
    printf("Max: %d \n", result - 1);

    cublasIsmin(handle, n, x, 1, &result);
    printf("Min: %d \n", result - 1);

    // Clean up the program
    cudaFree(x);
    cublasDestroy(handle);
    
    float result;

    return 0;
}