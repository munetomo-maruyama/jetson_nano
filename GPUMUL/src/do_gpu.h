//====================================================
// Calculate Square Matrix Multiplication
//     mul_gpu.h : Calculate by GPU (Header)
//----------------------------------------------------
// Rev.01 2019.05.05 M.Munetomo
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

#define SHARED_STATIC  (TRUE)
#define SHARED_DYNAMIC (!SHARED_STATIC)
//
#define BLOCK_SIZE 32
#define GPU_NITER 100

//----------------------------------
// Check Error during CUDA Runtime
//----------------------------------
#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

//------------------------------------
// Utility to access Shared Memeory
//------------------------------------
#if SHARED_DYNAMIC
#define sMat_A(x, y) sMat_AB[x + y * block_size_s]
#define sMat_B(x, y) sMat_AB[x + y * block_size_s + block_size_s * block_size_s]
#endif

//------------------------------------------------------------
// Calculate Square Matrix Multiplication using Global Memory
//------------------------------------------------------------
__global__ void GPU_G_Calc(float *dMat_A, float *dMat_B, float *dMat_G, uint32_t mat_size);

//--------------------------
// GPU Global Main Routine
//--------------------------
void GPU_G_Main(float *hMat_A, float *hMat_B, float *hMat_G, uint32_t mat_size);

//------------------------------------------------------------
// Calculate Square Matrix Multiplication using Shared Memory
//------------------------------------------------------------
__global__ void GPU_S_Calc(float *dMat_A, float *dMat_B, float *dMat_S, uint32_t mat_size);

//--------------------------
// GPU Shared Main Routine
//--------------------------
void GPU_S_Main(float *hMat_A, float *hMat_B, float *hMat_S, uint32_t mat_size);

//====================================================
// End of Program
//====================================================
