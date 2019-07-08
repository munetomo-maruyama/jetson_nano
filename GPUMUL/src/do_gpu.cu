//====================================================
// Calculate Square Matrix Multiplication
//     do_gpu.cu : Calculate by GPU (Program)
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

#include "common.h"
#include "do_gpu.h"

//------------------------------------------------------------
// Calculate Square Matrix Multiplication using Global Memory
//------------------------------------------------------------
__global__ void GPU_G_Calc(float *dMat_A, float *dMat_B, float *dMat_G, uint32_t mat_size)
{
    // Check Area
    uint32_t mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (mat_x >= mat_size) return;
    if (mat_y >= mat_size) return;
    //
    // Do Multiplication
    float element = 0.0f;
    for (uint32_t i = 0; i < mat_size; i++)
    {
        element = element
                + dMat_A[mat_y * mat_size + i] 
                * dMat_B[i * mat_size + mat_x];
    }
    dMat_G[mat_y * mat_size + mat_x] = element;
}

//--------------------------
// GPU Global Main Routine
//--------------------------
void GPU_G_Main(float *hMat_A, float *hMat_B, float *hMat_G, uint32_t mat_size)
{
    // Start Message
    printf("--------[GPU] Matrix (%d x %d) Multiplication using Global Memory ...\n", (int)mat_size, (int)mat_size);
    //
    // Allocate Buffers on Device
    float *dMat_A = NULL;
    float *dMat_B = NULL;
    float *dMat_G = NULL;
    int nBytes = sizeof(float) * mat_size * mat_size;
    CHECK(cudaMalloc((float **) &dMat_A, nBytes));
    CHECK(cudaMalloc((float **) &dMat_B, nBytes));
    CHECK(cudaMalloc((float **) &dMat_G, nBytes));
    CHECK(cudaMemcpy(dMat_A, hMat_A, nBytes, cudaMemcpyHostToDevice));        
    CHECK(cudaMemcpy(dMat_B, hMat_B, nBytes, cudaMemcpyHostToDevice));        
    //
    // Calculate Square Matrix Multiplication
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((mat_size + block.x - 1) / block.x, (mat_size + block.y - 1) / block.y); 
    printf("Grid = (%d, %d), Block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    //
    // Warm up
    GPU_G_Calc <<<grid, block>>> (dMat_A, dMat_B, dMat_G, mat_size);
    CHECK(cudaDeviceSynchronize());
    //
    // Do Actual Kernal    
    double iStart = CPU_Second();
    for (int n = 0; n < GPU_NITER; n++)
    {
        GPU_G_Calc <<<grid, block>>> (dMat_A, dMat_B, dMat_G, mat_size);
    }
    CHECK(cudaDeviceSynchronize());
    double iElaps = (CPU_Second() - iStart) / GPU_NITER;
    //
    // Display Result
    CHECK(cudaMemcpy(hMat_G, dMat_G, nBytes, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < mat_size * mat_size; i++)
    {
        if (i < MATRIX_SHOW) printf("A[%02d]=%8.4f B[%02d]=%8.4f G[%02d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_G[i]);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps, mat_size));
    //
    // Finish
    CHECK(cudaFree(dMat_A));
    CHECK(cudaFree(dMat_B));
    CHECK(cudaFree(dMat_G));
    CHECK(cudaDeviceReset());
}

//------------------------------------------------------------
// Calculate Square Matrix Multiplication using Shared Memory
//------------------------------------------------------------
// The mat_size should be multiple of BLOCK_SIZE.
#if SHARED_STATIC
__global__ void GPU_S_Calc(float *dMat_A, float *dMat_B, float *dMat_S, uint32_t mat_size)
#endif
#if SHARED_DYNAMIC
__global__ void GPU_S_Calc(float *dMat_A, float *dMat_B, float *dMat_S, uint32_t mat_size, uint32_t block_size_s)
#endif
{
    // Check Area
  //uint32_t mat_x = threadIdx.x + blockIdx.x * blockDim.x;
  //uint32_t mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    //
    // Allocate Shared Memory
    #if SHARED_STATIC    
    __shared__ float sMat_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sMat_B[BLOCK_SIZE][BLOCK_SIZE];
    #endif
    #if SHARED_DYNAMIC
    extern __shared__ float sMat_AB[]; // dynamic size allocation
    // sMat_A(x, y) --> sMat_AB[x + y * block_size_s]
    // sMat_B(x, y) --> sMat_AB[x + y * block_size_s + block_size_s * block_size_s]
    #endif
    //
    // Find Blocks to be processed
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    #if SHARED_STATIC
    uint32_t bgn_A = by * BLOCK_SIZE * mat_size;
    uint32_t end_A = bgn_A + mat_size - 1;
    uint32_t stp_A = BLOCK_SIZE;
    uint32_t bgn_B = bx * BLOCK_SIZE;
    uint32_t stp_B = BLOCK_SIZE * mat_size;
    #endif
    #if SHARED_DYNAMIC
    uint32_t bgn_A = by * block_size_s * mat_size;
    uint32_t end_A = bgn_A + mat_size - 1;
    uint32_t stp_A = block_size_s;
    uint32_t bgn_B = bx * block_size_s;
    uint32_t stp_B = block_size_s * mat_size;
    #endif
    //
    // Repeat for Sub Matrices; A:horizontal, B:Vertical
    uint32_t ia;
    uint32_t ib = bgn_B; 
    #if SHARED_DYNAMIC
  //uint32_t xA = tx;
  //uint32_t yA = ty + by * block_size_s;
  //uint32_t xB = tx + bx + block_size_s;
  //uint32_t yB = ty;
    #endif
    float element = 0.0f;
    for (ia = bgn_A; ia <= end_A;  ia = ia + stp_A)
    {
        // Load to Shared Memory from Global
        #if SHARED_STATIC
        sMat_A[ty][tx] = dMat_A[ia + ty * mat_size + tx];
        sMat_B[ty][tx] = dMat_B[ib + ty * mat_size + tx];
        #endif
        #if SHARED_DYNAMIC
        sMat_A(tx, ty) = dMat_A[ia + ty * mat_size + tx];
        sMat_B(tx, ty) = dMat_B[ib + ty * mat_size + tx];
      //if ((xA < mat_size) && (yA < mat_size))
      //{
      //    sMat_A(tx, ty) = dMat_A[ia + ty * mat_size + tx];
      //}
      //else
      //{
      //    sMat_A(tx, ty) = 0.0f;
      //}
      ////
      //if ((xB < mat_size) && (yB < mat_size))        
      //{
      //    sMat_B(tx, ty) = dMat_B[ib + ty * mat_size + tx];
      //}
      //else
      //{
      //    sMat_B(tx, ty) = 0.0f;
      //}
      //xA = xA + block_size_s;
      //yB = yB + block_size_s;
        #endif
        //
        // Wait for all threads finish loading into same shared memory
        __syncthreads();
        //
        // Do Multiplication
        #if SHARED_STATIC
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
        {
            element = element + sMat_A[ty][i] * sMat_B[i][tx];
        }        
        #endif
        #if SHARED_DYNAMIC
        for (uint32_t i = 0; i < block_size_s; i++)
        {
          //if (bx * block_size_s + i >= mat_size) continue;
          //if (by * block_size_s + i >= mat_size) continue;
            //
            element = element + sMat_A(i, ty) * sMat_B(tx, i);
        }
        #endif        
        //
        // Wait for all threads finish calculating
        __syncthreads();
        //
        // Next
        ib = ib + stp_B;
    }
    //
    // Store the result
    #if SHARED_STATIC
    uint32_t bgn_S = by * BLOCK_SIZE * mat_size + bx * BLOCK_SIZE;
    uint32_t idx_S = bgn_S + ty * mat_size + tx;
    dMat_S[idx_S] = element;
    #endif
    #if SHARED_DYNAMIC
    uint32_t bgn_S = by * block_size_s * mat_size + bx * block_size_s;
    uint32_t idx_S = bgn_S + ty * mat_size + tx;
    dMat_S[idx_S] = element;
  //if ((mat_x < mat_size) && (mat_y < mat_size))
  //{
  //    uint32_t bgn_S = by * block_size_s * mat_size + bx * block_size_s;
  //    uint32_t idx_S = bgn_S + ty * mat_size + tx;
  //    dMat_S[idx_S] = element;
  //}
    #endif
}

//--------------------------
// GPU Shared Main Routine
//--------------------------
void GPU_S_Main(float *hMat_A, float *hMat_B, float *hMat_S, uint32_t mat_size)
{
    // Start Message
    printf("--------[GPU] Matrix (%d x %d) Multiplication using Shared Memory ...\n", (int)mat_size, (int)mat_size);
    //
    // Re-size Matrix to multiple of BLOCK_SIZE
    uint32_t mat_resize = ((mat_size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
    int nBytes = sizeof(float) * mat_resize * mat_resize;
    //
    // Allocate Buffers
    float *hMat_Ar, *hMat_Br, *hMat_Sr;
    if ((hMat_Ar = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_Br = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_Sr = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    //
    // Copy Contents to resized buffer
    for (uint32_t y = 0; y < mat_resize; y++)
    {
        for (uint32_t x = 0; x < mat_resize; x++)
        {
            hMat_Ar[x + y * mat_resize] = ((x < mat_size) && (y < mat_size))? hMat_A[x + y * mat_size] : 0.0f;
            hMat_Br[x + y * mat_resize] = ((x < mat_size) && (y < mat_size))? hMat_B[x + y * mat_size] : 0.0f;
        }
    }    
    //
    // Allocate Buffers on Device
    float *dMat_Ar = NULL;
    float *dMat_Br = NULL;
    float *dMat_Sr = NULL;
    CHECK(cudaMalloc((float **) &dMat_Ar, nBytes));
    CHECK(cudaMalloc((float **) &dMat_Br, nBytes));
    CHECK(cudaMalloc((float **) &dMat_Sr, nBytes));
    CHECK(cudaMemcpy(dMat_Ar, hMat_Ar, nBytes, cudaMemcpyHostToDevice));        
    CHECK(cudaMemcpy(dMat_Br, hMat_Br, nBytes, cudaMemcpyHostToDevice));        
    //
    // Calculate Square Matrix Mutiplication
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((mat_resize + block.x - 1) / block.x, (mat_resize + block.y - 1) / block.y); 
    printf("Grid = (%d, %d), Block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    #if SHARED_DYNAMIC
    uint32_t shared_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    uint32_t block_size_s = BLOCK_SIZE;
    #endif
    //
    // Warm up
    #if SHARED_STATIC
    GPU_S_Calc <<<grid, block>>> (dMat_Ar, dMat_Br, dMat_Sr, mat_resize);
    #endif
    #if SHARED_DYNAMIC
    GPU_S_Calc<<<grid, block, shared_size>>>(dMat_Ar, dMat_Br, dMat_Sr, mat_resize, block_size_s);
    #endif
    CHECK(cudaDeviceSynchronize());
    //
    // Do Actual Kernel    
  //cudaEvent_t start, stop;
  //CHECK(cudaEventCreate(&start));
  //CHECK(cudaEventCreate(&stop));
  //CHECK(cudaEventRecord(start, NULL));
    double iStart = CPU_Second();
    for (int n = 0; n < GPU_NITER; n++)
    {
        #if SHARED_STATIC
        GPU_S_Calc <<<grid, block>>> (dMat_Ar, dMat_Br, dMat_Sr, mat_resize);
        #endif
        #if SHARED_DYNAMIC
        GPU_S_Calc<<<grid, block, shared_size>>>(dMat_Ar, dMat_Br, dMat_Sr, mat_resize, block_size_s);
        #endif
    }
  //CHECK(cudaEventRecord(stop, NULL));    
  //CHECK(cudaEventSynchronize(stop));
  //float msec;
  //CHECK(cudaEventElapsedTime(&msec, start, stop));
  //double iElaps = (double)(msec * 1.0e-3 / GPU_NITER);
    CHECK(cudaDeviceSynchronize());
    double iElaps = (CPU_Second() - iStart) / GPU_NITER;
    //
    // Copy back Result
    CHECK(cudaMemcpy(hMat_Sr, dMat_Sr, nBytes, cudaMemcpyDeviceToHost));
    for (uint32_t y = 0; y < mat_size; y++)
    {
        for (uint32_t x = 0; x < mat_size; x++)
        {
            hMat_S[x + y * mat_size] = hMat_Sr[x + y * mat_resize];
        }
    }        
    //
    // Display Result
    for (uint32_t i = 0; i < mat_size * mat_size; i++)
    {
        if (i < MATRIX_SHOW) printf("A[%02d]=%8.4f B[%02d]=%8.4f S[%02d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_S[i]);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps, mat_size));
    //
    // Finish
    CHECK(cudaFree(dMat_Ar));
    CHECK(cudaFree(dMat_Br));
    CHECK(cudaFree(dMat_Sr));
    if (hMat_Ar) free(hMat_Ar);
    if (hMat_Br) free(hMat_Br);
    if (hMat_Sr) free(hMat_Sr);
    CHECK(cudaDeviceReset());
}

//====================================================
// End of Program
//====================================================
