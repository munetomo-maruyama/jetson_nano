//====================================================
// Calculate Matrix Addition
//     mul_gpu.h : Calculate by GPU (Header)
//----------------------------------------------------
// Rev.01 2019.06.29 M.Munetomo
//----------------------------------------------------
// Copyright (C) 2019 Munetomo Maruyama
//====================================================

#include <cinttypes>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32
#define GPU_NITER 10

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

//---------------------------
// Calculate Matrix Addition 
//---------------------------
__global__ void GPU_Calc(float *dMat_A, float *dMat_B, float *dMat_G, uint32_t mat_size_x, uint32_t mat_size_y);

//--------------------------
// GPU Main Routine
//--------------------------
void GPU_Main(float *hMat_A, float *hMat_B, float *hMat_G, uint32_t mat_size_x, uint32_t mat_size_y);

//====================================================
// End of Program
//====================================================
