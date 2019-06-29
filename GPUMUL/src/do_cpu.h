//====================================================
// Calculate Square Matrix Multiplication
//     mul_cpu.h : Calculate by CPU (Header)
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

//---------------------------------------
// Calculate Square Matrix Mutiplication
//---------------------------------------
void CPU_Calc(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size);

//--------------------
// CPU Main Routine
//--------------------
void CPU_Main(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size);

//--------------------
// Compare Matrices
//--------------------
void Compare(float *ref, float *tgt, uint32_t mat_size, char *str);

//====================================================
// End of Program
//====================================================
