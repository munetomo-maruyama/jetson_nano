//====================================================
// Calculate Matrix Addition
//     do_gpu.cu : Calculate by GPU (Program)
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

#include "common.h"
#include "do_gpu.h"

//---------------------------
// Calculate Matrix Addition 
//---------------------------
__global__ void GPU_Calc(float *dMat_A, float *dMat_B, float *dMat_G, uint32_t mat_size_x, uint32_t mat_size_y)
{
    // Check Area
    uint32_t mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (mat_x >= mat_size_x) return;
    if (mat_y >= mat_size_y) return;
    //
    // Do Addition
    uint32_t index = mat_y * mat_size_x + mat_x;
    dMat_G[index] = dMat_A[index] + dMat_B[index];
}

//-------------------
// GPU Main Routine
//-------------------
void GPU_Main(float *hMat_A, float *hMat_B, float *hMat_G, uint32_t mat_size_x, uint32_t mat_size_y)
{
    // Start Message
    printf("--------[GPU] Matrix (%d x %d) Addition ...\n", (int)mat_size_x, (int)mat_size_y);
    //
    // Allocate Buffers on Device
    float *dMat_A = NULL;
    float *dMat_B = NULL;
    float *dMat_G = NULL;
    int nBytes = sizeof(float) * mat_size_x * mat_size_y;
    CHECK(cudaMalloc((float **) &dMat_A, nBytes));
    CHECK(cudaMalloc((float **) &dMat_B, nBytes));
    CHECK(cudaMalloc((float **) &dMat_G, nBytes));
    CHECK(cudaMemcpy(dMat_A, hMat_A, nBytes, cudaMemcpyHostToDevice));        
    CHECK(cudaMemcpy(dMat_B, hMat_B, nBytes, cudaMemcpyHostToDevice));        
    //
    // Calculate Square Matrix Multiplication
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((mat_size_x + block.x - 1) / block.x, (mat_size_y + block.y - 1) / block.y); 
    printf("Grid = (%d, %d), Block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    //
    // Warm up
    GPU_Calc <<<grid, block>>> (dMat_A, dMat_B, dMat_G, mat_size_x, mat_size_y);
    CHECK(cudaDeviceSynchronize());
    //
    // Do Actual Kernal    
    double iStart = CPU_Second();
    for (int n = 0; n < GPU_NITER; n++)
    {
        GPU_Calc <<<grid, block>>> (dMat_A, dMat_B, dMat_G, mat_size_x, mat_size_y);
    }
    CHECK(cudaDeviceSynchronize());
    double iElaps = (CPU_Second() - iStart) / GPU_NITER;
    //
    // Display Result
    CHECK(cudaMemcpy(hMat_G, dMat_G, nBytes, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++)
    {
        if (i < MATRIX_SHOW) printf("A[%02d]=%8.4f B[%02d]=%8.4f G[%02d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_G[i]);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps, mat_size_x, mat_size_y));
    //
    // Finish
    CHECK(cudaFree(dMat_A));
    CHECK(cudaFree(dMat_B));
    CHECK(cudaFree(dMat_G));
    CHECK(cudaDeviceReset());
}

//====================================================
// End of Program
//====================================================
