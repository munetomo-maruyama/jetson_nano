//====================================================
// Device Recognization
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.06.29 M.Munetomo
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

#define DATA_SIZE_X 64
#define DATA_SIZE_Y 32
#define BLOCK_SIZE (8, 4)

//-----------------
// Device Kernel
//-----------------
__global__ void Device_Kernel(void)
{
	uint32_t ix = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t iy = threadIdx.y + blockIdx.y * blockDim.y;
	printf("Device (ix, iy) = (%d, %d) : \
threadIdx.x = %d, blockIdx.x = %d, blockDim.x = %d, \
threadIdx.y = %d, blockIdx.y = %d, blockDim.y = %d\n",
	        ix, iy, 
	        threadIdx.x, blockIdx.x, blockDim.x,
	        threadIdx.y, blockIdx.y, blockDim.y);
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

//----------------------------------
// Main Routine
//----------------------------------
int main(void)
{
	// Grids and Blocks
    dim3 block BLOCK_SIZE;
    dim3 grid(((DATA_SIZE_X) + block.x - 1) / block.x, 
              ((DATA_SIZE_Y) + block.y - 1) / block.y); 
    //
    // Call Kernel (warm up)
    Device_Kernel <<<grid, block>>> ();
    //
    // Wait for termination of all threads
    CHECK(cudaDeviceSynchronize());
    //
    // Reset Device
    CHECK(cudaDeviceReset());
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
