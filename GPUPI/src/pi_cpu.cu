//====================================================
// Calculate any digit of Pi using BBP formula
//     pi_cpu.cu : Calculate Pi by CPU (Program)
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
#include "pi_cpu.h"

//----------------------------------------------
// Binary Modulo Exponentiation r=(a^b)mod(c)
//----------------------------------------------
int64_t CPU_Bin_Mod_Exp(int64_t a, int64_t b, int64_t c)
{
  //printf("binmodexp(%ld,%ld,%ld)", a,b,c);
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
  //printf("=%ld\n", ret);
	return ret;
}

//------------------------------
// Calculate S(j, d)
//------------------------------
double CPU_Sjd(int64_t j, int64_t d)
{
    double sum = 0;
    // k = 0 ... d
    for (int64_t k = 0; k <= d; k++)
    {
        sum = sum + (double)CPU_Bin_Mod_Exp(16, d - k, 8 * k + j) / (double)(8 * k + j);
    }
    // k = (d + 1) ...
  //double error = 1 / (double) (0x1000000000);
    double numerator = 1;
    double denominator = 8 * d + j;
    double increase;
    for (int64_t k = 0; k < 8; k++)
    {
        numerator = numerator / 16;
        denominator = denominator + 8;
        increase = numerator / denominator;
        sum = sum + increase;
      //if (increase < error) break;
      //printf(".");
    }
    //    
    sum = sum - (int)sum;  // extract decimal part
  //printf("S(%ld,%ld)=%lf\n", j, d, sum);
    //
    return sum;
}

//-----------------------------
// Whole Number to Hex
//-----------------------------
char CPU_WholeNum_to_Hex(double input)
{
    const char hex_table[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
    int index = (int) input;
    if ((index < 0) || (index > 15)) return '*';
    return hex_table[index];
}

//------------------------------------------
// Calculate Pi at specified digit
//------------------------------------------
void CPU_Calc_Pi(int64_t digit, char *result_hex)
{
    double Pi16d = 0;
    //
	Pi16d = 4 * CPU_Sjd(1, digit)
          - 2 * CPU_Sjd(4, digit)
          - 1 * CPU_Sjd(5, digit)
          - 1 * CPU_Sjd(6, digit);
    //
	Pi16d = (Pi16d > 0) ? (Pi16d - (int) Pi16d) : (Pi16d - (int) Pi16d + 1);
    //
    for (int i = 0; i < DIGIT_STEP; i++)
    {
        Pi16d = Pi16d * 16;
        *(result_hex + digit + i) = CPU_WholeNum_to_Hex(Pi16d);
        Pi16d = Pi16d - (int) Pi16d;
    }
}

//====================================================
// End of Program
//====================================================
