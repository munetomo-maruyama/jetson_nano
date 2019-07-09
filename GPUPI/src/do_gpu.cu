//====================================================
// Calculate any digit of Pi using BBP formula
//     pi_gpu.cu : Calculate Pi by GPU (Program)
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

#include "common.h"
#include "do_gpu.h"

//--------------------------------
// Constant Hex Table
//--------------------------------
__constant__ char CM_hex_table[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

//----------------------------------------------
// Binary Modulo Exponentiation r=(a^b)mod(c)
//----------------------------------------------
__device__ int64_t GPU_Bin_Mod_Exp(int64_t a, int64_t b, int64_t c)
{
    int64_t ret = 1;
    while (b != 0)
    {
        if (b % 2)
        {
            ret = (ret * a) % c;
        }
        a = (a * a) % c;
        b = b / 2;
    }
    return ret;
}

//------------------------------
// Calculate S(j, d)
//------------------------------
__device__ double GPU_Sjd(int64_t j, int64_t d)
{
    double sum = 0;
    // k = 0 ... d
    for (int64_t k = 0; k <= d; k++)
    {
        sum = sum + (double)GPU_Bin_Mod_Exp(16, d - k, 8 * k + j) / (double)(8 * k + j);
    }
    // k = (d + 1) ...
    double numerator = 1;
    double denominator = 8 * d + j;
    double increase;
    for (int64_t k = 0; k < 8; k++)
    {
        numerator = numerator / 16;
        denominator = denominator + 8;
        increase = numerator / denominator;
        sum = sum + increase;
    }
    //    
    sum = sum - (int)sum;  // extract decimal part
    //
    return sum;
}

//-----------------------------
// Whole Number to Hex
//-----------------------------
__device__ char GPU_WholeNum_to_Hex(double input)
{
    int index = (int) input;
    if ((index < 0) || (index > 15)) return '*';
    return CM_hex_table[index];
}

//------------------------------------------
// Calculate Pi at specified digit
//------------------------------------------
__global__ void GPU_Calc_Pi_kernel(uint64_t digit_max, char *result_hex)
{
    // Which digit should I calculate?
    uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= digit_max / DIGIT_STEP) return;
    int64_t digit = ix * DIGIT_STEP;
    //
    double Pi16d = 0;
    //
    Pi16d = 4 * GPU_Sjd(1, digit)
          - 2 * GPU_Sjd(4, digit)
          - 1 * GPU_Sjd(5, digit)
          - 1 * GPU_Sjd(6, digit);
    //
    Pi16d = (Pi16d > 0) ? (Pi16d - (int) Pi16d) : (Pi16d - (int) Pi16d + 1);
    //
    for (int i = 0; i < DIGIT_STEP; i++)
    {
        Pi16d = Pi16d * 16;
        *(result_hex + digit + i) = GPU_WholeNum_to_Hex(Pi16d);
        Pi16d = Pi16d - (int) Pi16d;
    }
}

//--------------------------------
// GPU Main Routine
//--------------------------------
void GPU_Main(int64_t digit)
{
    //
    // Start Message
    printf("--------[GPU] %lld digits Calculation...(showing specified last %d digits)\n", (long long int)digit, DIGIT_SHOW);
    //
    // Adjust Digits
    int64_t digit_show_bgn = (digit < DIGIT_SHOW)? 0 : digit -DIGIT_SHOW; // digit to be shown
    digit = ((digit + DIGIT_STEP - 1) / DIGIT_STEP) * DIGIT_STEP; // digit to be calculated
    //
    // Allocate Result Buffer
    char *h_gpu_result = NULL;
    h_gpu_result = (char*) malloc((digit + DIGIT_STEP) * sizeof(char));
    if (h_gpu_result == NULL) exit(EXIT_FAILURE);
    memset(h_gpu_result, 0, digit + DIGIT_STEP);
    //
    char *d_gpu_result = NULL;
    CHECK(cudaMalloc((void **) &d_gpu_result, (digit + DIGIT_STEP) * sizeof(char)));
    CHECK(cudaMemcpy(d_gpu_result, h_gpu_result, digit, cudaMemcpyHostToDevice));        
    //
    // Calculate Pi by BBP formula
    double iStart = CPU_Second();
    dim3 block(256);
    dim3 grid((digit / DIGIT_STEP + block.x - 1) / block.x);
    GPU_Calc_Pi_kernel<<<grid, block>>>(digit, d_gpu_result);
    CHECK(cudaDeviceSynchronize());
    double iElaps = CPU_Second() - iStart;
    //
    // Display Result
    CHECK(cudaMemcpy(h_gpu_result, d_gpu_result, digit, cudaMemcpyDeviceToHost));
    int64_t d;
    for (d = digit_show_bgn; d < digit_show_bgn + DIGIT_SHOW; d++)
    {
        printf("%c", h_gpu_result[d]);
    }
    printf("\n");
    printf("Time elapsed %lf sec\n", iElaps);    
    //
    // Finish
    CHECK(cudaFree(d_gpu_result));
    CHECK(cudaDeviceReset());
    free(h_gpu_result);
}

//====================================================
// End of Program
//====================================================
