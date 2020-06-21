#include <stdio.h>
#include "cublas_v2.h"
#include "cuda_runtime.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define m 5
#define n 4

int main() {

    // A matrix
    float* a;
    cudaMallocManaged(&a, m * n * sizeof(float));
    int ind=11;
    for(int j=0;j<n;j++)
        for(int i=0;i<m;i++)
            a[IDX2C(i,j,m)] = (float)ind++;

    // X Vector
    float* x;
    cudaMallocManaged(&x, n * sizeof(float));
    for(int i=0;i<n;i++) x[i] = 1.0f;

    // Y Vector
    float* y;
    cudaMallocManaged(&y, m * sizeof(float));
    for(int j=0;j<m;j++) y[j] = 0.0f;

    // Handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // y = alpha*a*x + b*y
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, a, m, x, 1, &beta, y, 1);
    cudaDeviceSynchronize();

    printf("Y: ");
    for(int i=0;i<m;i++)
        printf(" %0.5f ", y[i]);
    printf("\n");

    // Clean up the program
    cudaFree(a);
    cudaFree(x);
    cudaFree(y);
    cublasDestroy(handle);
    return 0;
}