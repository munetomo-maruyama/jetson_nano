//====================================================
// Calculate Square Matrix Multiplication
//     do_cpu.cu : Calculate by CPU (Program)
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
#include <math.h>

#include "common.h"
#include "do_cpu.h"

//---------------------------------------
// Calculate Square Matrix Mutiplication
//---------------------------------------
void CPU_Calc_Mat_Mul(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size)
{
    for (uint32_t y = 0; y < mat_size; y++)
    {
        for (uint32_t x = 0; x < mat_size; x++)
        {
            float element = 0.0f;
            for (uint32_t i = 0; i < mat_size; i++)
            {
                element = element
                        + hMat_A[y * mat_size + i] 
                        * hMat_B[i * mat_size + x];
            }
            hMat_C[y * mat_size + x] = element;
        }
    }
}

//--------------------
// CPU Main Routine
//--------------------
void CPU_Main(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size)
{
    // Start Message
    printf("--------[CPU] Matrix (%d x %d) Multiplication...\n", (int)mat_size, (int)mat_size);
    //
    // Calculate Square Matrix Mutiplication
    double iStart = CPU_Second();
    CPU_Calc_Mat_Mul(hMat_A, hMat_B, hMat_C, mat_size);
    double iElaps = CPU_Second() - iStart;
    //
    // Display Result
    for (uint32_t i = 0; i < mat_size * mat_size; i++)
    {
        if (i < MATRIX_SHOW) printf("A[%02d]=%8.4f B[%02d]=%8.4f C[%02d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_C[i]);
    }
    printf("Time elapsed %lf sec (%lf GFLOPS)\n", iElaps, GFLOPS(iElaps, mat_size));
}

//--------------------
// Compare Matrices
//--------------------
void Compare(float *ref, float *tgt, uint32_t mat_size, char *str)
{
    int unmatch = 0;
    float error = 1.0e-8;
    uint32_t i;
    for (i = 0; i < mat_size * mat_size; i++)
    {
        if (fabsf(ref[i] - tgt[i]) > error)
        {
            unmatch = 1;
            break;
        }
    }
    if (!unmatch) printf("%s : Matched !\n", str);
    if ( unmatch) printf("%s : Unmatched...! (%d %f %f)\n", str, i, ref[i], tgt[i]);    
}

//====================================================
// End of Program
//====================================================
