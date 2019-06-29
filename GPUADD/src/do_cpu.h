//====================================================
// Calculate Matrix Addition
//     mul_cpu.h : Calculate by CPU (Header)
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

//---------------------------------------
// Calculate Matrix Addition
//---------------------------------------
void CPU_Calc(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y);

//--------------------
// CPU Main Routine
//--------------------
void CPU_Main(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y);

//--------------------
// Compare Matrices
//--------------------
void Compare(float *ref, float *tgt, uint32_t mat_size_x, uint32_t mat_size_y);

//====================================================
// End of Program
//====================================================
