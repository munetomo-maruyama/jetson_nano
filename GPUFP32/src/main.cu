//====================================================
// Towards Maximum GFLOPS
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.05.11 M.Munetomo
//----------------------------------------------------
// Copyright (C) 2019 Munetomo Maruyama
//====================================================

#include <cinttypes>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define DATA_SIZE (1024*1024)
#define ITERATION  65536
#define BLOCK_SIZE 1024
#define COEFF_A 0.4999
#define COEFF_B 1.2345

//-----------------
// Device Kernel
//-----------------
__global__ void Device_Kernel(float *buf, const float a, const float b)
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= DATA_SIZE) return;
    //
    float c = buf[index];
    //
    for (int i = 0; i < ITERATION; i++)
    {
         c = a * c + b;
    }
    buf[index] = c;
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
    double operations = (double)ITERATION * (double)DATA_SIZE * 2.0;
    double gflops = operations * 1.0e-9f / sec;
    return gflops;
}

//----------------------------------
// Main Routine
//----------------------------------
int main(void)
{
    // Allocate Host Buffer
    float *hBuf;
    if ((hBuf = (float*)malloc(sizeof(float) * DATA_SIZE)) == NULL) exit(EXIT_FAILURE);
    //
    // Generate Random Data
    time_t t;
    srand((unsigned int)time(&t));
    for (uint32_t i = 0; i < DATA_SIZE; i++)
    {
        hBuf[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
    }
    //
    // Allocate Device Buffer
    float *dBuf;
    CHECK(cudaMalloc((float **) &dBuf, sizeof(float) * DATA_SIZE));
    CHECK(cudaMemcpy(dBuf, hBuf, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));
    //
    // Grids and Blocks
    dim3 block(BLOCK_SIZE);
    dim3 grid(((DATA_SIZE) + block.x - 1) / block.x); 
    //
    // Call Kernel (warm up)
    Device_Kernel <<<grid, block>>> (dBuf, COEFF_A, COEFF_B);
    CHECK(cudaDeviceSynchronize());
    //
    // Call Kernel (measure)
    double iStart = CPU_Second();
    Device_Kernel <<<grid, block>>> (dBuf, COEFF_A, COEFF_B);
    CHECK(cudaDeviceSynchronize());
    double iElaps = CPU_Second() - iStart;
    //
    // Display Result
    CHECK(cudaMemcpy(hBuf, dBuf, sizeof(float) * DATA_SIZE, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < 10; i++)
    {
        printf("hBuf[%02d]=%8.4f\n", i, hBuf[i]);
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
