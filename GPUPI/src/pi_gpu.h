//====================================================
// Calculate any digit of Pi using BBP formula
//     pi_gpu.h : Calculate Pi by GPU (Header)
//----------------------------------------------------
// Rev.01 2019.04.28 M.Munetomo
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

//----------------------------------------------
// Binary Modulo Exponentiation r=(a^b)mod(c)
//----------------------------------------------
__device__ int64_t GPU_Bin_Mod_Exp(int64_t a, int64_t b, int64_t c);

//------------------------------
// Calculate S(j, d)
//------------------------------
__device__ double GPU_Sjd(int64_t j, int64_t d);

//-----------------------------
// Whole Number to Hex
//-----------------------------
__device__  char GPU_WholeNum_to_Hex(double input);

//------------------------------------------
// Calculate Pi at specified digit
//------------------------------------------
__global__ void GPU_Calc_Pi_kernel(uint64_t digit_max, char *result_hex);

//====================================================
// End of Program
//====================================================
