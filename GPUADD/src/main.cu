//====================================================
// Calculate Matrix Addition
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

#include "common.h"
#include "do_cpu.h"
#include "do_gpu.h"

//-----------------
// Print Usage
//-----------------
void print_usage(char *cmd)
{
    printf("[Usage] %s [-c] [-g] matrix_size_x matrix_size_y\n", cmd);
    printf("    -c : Invoke CPU\n"); 
    printf("    -g : Invoke GPU\n"); 
    printf("    matrix_size_x : Specify Matrix Size X\n");
    printf("    matrix_size_y : Specify Matrix Size Y\n");
    exit(EXIT_FAILURE);
}

//-----------------
// CPU Time
//-----------------
double CPU_Second(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//------------------------
// Calculate Giga FLOPS
//------------------------
double GFLOPS(double sec, uint32_t mat_size_x, uint32_t mat_size_y)
{
    double operations = mat_size_x * mat_size_y;
    double gflops = operations * 1.0e-9f / sec;
    return gflops;
}

//----------------------------------
// Main Routine
//----------------------------------
int main(int argc, char *argv[])
{
    //
    // Get Arguments
    int opt;
    int do_cpu = 0;
    int do_gpu = 0;
    opterr = 0;    
    while ((opt = getopt(argc, argv, "cg")) != -1)
    {
        switch (opt)
        {
            case 'c' : {do_cpu = 1; break;}
            case 'g' : {do_gpu = 1; break;}
            default  : {print_usage(argv[0]); break;}
        }
    }
    argc = argc - optind;
    if (argc < 2) print_usage(argv[0]);
    argv = argv + optind;
    uint32_t mat_size_x = (uint32_t) atol(argv[0]);
    uint32_t mat_size_y = (uint32_t) atol(argv[1]);
    //
    // Allocate Buffers in Host
    int nBytes = sizeof(float) * mat_size_x * mat_size_y;
    float *hMat_A, *hMat_B, *hMat_C, *hMat_G;
    if ((hMat_A = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_B = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_C = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_G = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    //
    // Generate Random Matrix
    time_t t;
    srand((unsigned int)time(&t));
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++)
    {
        hMat_A[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
        hMat_B[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
    }
    //
    // Do each Calculation
    if (do_cpu) CPU_Main(hMat_A, hMat_B, hMat_C, mat_size_x, mat_size_y);
    if (do_gpu) GPU_Main(hMat_A, hMat_B, hMat_G, mat_size_x, mat_size_y);
    //
    // Do Verification if possible
    if (do_cpu && do_gpu) Compare(hMat_C, hMat_G, mat_size_x, mat_size_y);
    //
    // Finish
    if (hMat_A) free(hMat_A);
    if (hMat_B) free(hMat_B);
    if (hMat_C) free(hMat_C);
    if (hMat_G) free(hMat_G);
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
