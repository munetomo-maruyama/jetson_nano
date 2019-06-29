//====================================================
// Calculate Matrix Addition
//     do_cpu.cu : Calculate by CPU (Program)
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
#include <math.h>

#include "common.h"
#include "do_cpu.h"

//---------------------------------------
// Calculate Matrix Addition
//---------------------------------------
void CPU_Calc(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y)
{
    for (uint32_t y = 0; y < mat_size_y; y++)
    {
        for (uint32_t x = 0; x < mat_size_x; x++)
        {
			uint32_t index = y * mat_size_x + x;
			hMat_C[index] = hMat_A[index] + hMat_B[index];
        }
    }
}

//--------------------
// CPU Main Routine
//--------------------
void CPU_Main(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y)
{
    // Start Message
    printf("--------[CPU] Matrix (%d x %d) Addition...\n", (int)mat_size_x, (int)mat_size_y);
    //
    // Calculate Matrix Addition
    double iStart = CPU_Second();
    CPU_Calc(hMat_A, hMat_B, hMat_C, mat_size_x, mat_size_y);
    double iElaps = CPU_Second() - iStart;
    //
    // Display Result
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++)
    {
        if (i < MATRIX_SHOW) printf("A[%02d]=%8.4f B[%02d]=%8.4f C[%02d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_C[i]);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps, mat_size_x, mat_size_y));
}

//--------------------
// Compare Matrices
//--------------------
void Compare(float *ref, float *tgt, uint32_t mat_size_x, uint32_t mat_size_y)
{
    int unmatch = 0;
    float error = 1.0e-8;
    uint32_t i;
    for (i = 0; i < mat_size_x * mat_size_y; i++)
    {
        if (fabsf(ref[i] - tgt[i]) > error)
        {
            unmatch = 1;
            break;
        }
    }
    if (!unmatch) printf("Matched !\n");
    if ( unmatch) printf("Unmatched...! (%d %f %f)\n", i, ref[i], tgt[i]);    
}

//====================================================
// End of Program
//====================================================
