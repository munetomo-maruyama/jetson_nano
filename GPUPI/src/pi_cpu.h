//====================================================
// Calculate any digit of Pi using BBP formula
//     pi_cpu.h : Calculate Pi by CPU (Header)
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

#define DIGIT_STEP 8

//----------------------------------------------
// Binary Modulo Exponentiation r=(a^b)mod(c)
//----------------------------------------------
int64_t CPU_Bin_Mod_Exp(int64_t a, int64_t b, int64_t c);

//------------------------------
// Calculate S(j, d)
//------------------------------
double CPU_Sjd(int64_t j, int64_t d);

//-----------------------------
// Whole Number to Hex
//-----------------------------
char CPU_WholeNum_to_Hex(double input);

//------------------------------------------
// Calculate Pi at specified digit
//------------------------------------------
void CPU_Calc_Pi(int64_t digit, char *result_hex);

//====================================================
// End of Program
//====================================================
