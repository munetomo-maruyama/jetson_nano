//====================================================
// Towards Maximum GFLOPS
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.05.11 M.Munetomo
//----------------------------------------------------
// Copyright (C) 2019 Munetomo Maruyama
//====================================================

#include <cinttypes>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define DATA_SIZE (1024*1024)
#define DATA_SIZE_H (DATA_SIZE / 2)
#define ITERATION  65536
#define BLOCK_SIZE 1024
#define COEFF_A 0.4999
#define COEFF_B 1.2345

//-----------------
// Device Kernel
//-----------------
__global__ void Device_Kernel(half2 *buf)
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= DATA_SIZE_H) return;
    //
    half2 x = buf[index];
    const half2 a = __floats2half2_rn(COEFF_A, COEFF_A);
    const half2 b = __floats2half2_rn(COEFF_B, COEFF_B);
    //
    for (int i = 0; i < ITERATION; i++)
    {
        x = a * x + b;
    }
    buf[index] = x;
}

//----------------------------------
// Check Error during CUDA Runtime
//----------------------------------
#define CHECK(func)                                    \
{                                                      \
    const cudaError_t error = func;                    \
    if (error != cudaSuccess)                          \
    {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);  \
        printf("Code:%d, Reason: %s\n", error,         \
                cudaGetErrorString(error));            \
        cudaDeviceReset();                             \
        exit(EXIT_FAILURE);                            \
    }                                                  \
}

//-----------------
// CPU Time
//-----------------
double CPU_Second(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//------------------------
// Calculate Giga FLOPS
//------------------------
double GFLOPS(double sec)
{
    double operations = (double)ITERATION * (double)DATA_SIZE * 2;
    double gflops = operations * 1.0e-9f / sec;
    return gflops;
}

//----------------------------------
// Main Routine
//----------------------------------
int main(void)
{
    // Allocate Host Buffer
    half2 *hBuf;
    if ((hBuf = (half2*)malloc(sizeof(half2) * DATA_SIZE_H)) == NULL) exit(EXIT_FAILURE);
    //
    // Generate Random Data
    time_t t;
    srand((unsigned int)time(&t));
    for (uint32_t i = 0; i < DATA_SIZE_H; i++)
    {
        float fL = (float)((rand() % 10000) - 5000) / 1000.0f; // low
        float fH = (float)((rand() % 10000) - 5000) / 1000.0f; // high
        hBuf[i] = __floats2half2_rn(fL, fH);
    }
    //
    // Allocate Device Buffer
    half2 *dBuf;
    CHECK(cudaMalloc((half2 **) &dBuf, sizeof(half2) * DATA_SIZE_H));
    CHECK(cudaMemcpy(dBuf, hBuf, sizeof(half2) * DATA_SIZE_H, cudaMemcpyHostToDevice));
    //
    // Grids and Blocks
    dim3 block(BLOCK_SIZE);
    dim3 grid(((DATA_SIZE_H) + block.x - 1) / block.x); 
    //
    // Call Kernel (warm up)
    Device_Kernel <<<grid, block>>> (dBuf);
    CHECK(cudaDeviceSynchronize());
    //
    // Call Kernel (measure)
    double iStart = CPU_Second();
    Device_Kernel <<<grid, block>>> (dBuf);
    CHECK(cudaDeviceSynchronize());
    double iElaps = CPU_Second() - iStart;
    //
    // Display Result
    CHECK(cudaMemcpy(hBuf, dBuf, sizeof(half2) * DATA_SIZE_H, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < 10; i = i + 2)
    {
        float fL = __low2float(hBuf[i]);  // low
        float fH = __high2float(hBuf[i]); // high
        printf("hBuf[%02d]=%8.4f\n", i + 0, fL);
        printf("hBuf[%02d]=%8.4f\n", i + 1, fH);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps));
    //
    // Finish
    CHECK(cudaFree(dBuf));
    if (hBuf) free(hBuf);
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
