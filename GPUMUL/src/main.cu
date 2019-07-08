//====================================================
// Calculate Square Matrix Multiplication
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.05.05 M.Munetomo
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
    printf("[Usage] %s [-c] [-g] [-s] matrix_size\n", cmd);
    printf("    -c : Invoke CPU\n"); 
    printf("    -g : Invoke GPU using Global Memory with block_size\n"); 
    printf("    -s : Invoke GPU using Shared Memory with block_size\n"); 
    printf("    matrix_size : Specify Square Matrix Size\n");
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
double GFLOPS(double sec, uint32_t mat_size)
{
    double operations = 2.0 * mat_size * mat_size * mat_size;
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
    int do_gpu_global = 0;
    int do_gpu_shared = 0;
    opterr = 0;    
    while ((opt = getopt(argc, argv, "cgs")) != -1)
    {
        switch (opt)
        {
            case 'c' : {do_cpu = 1; break;}
            case 'g' : {do_gpu_global = 1; break;}
            case 's' : {do_gpu_shared = 1; break;}
            default  : {print_usage(argv[0]); break;}
        }
    }
    argc = argc - optind;
    if (argc < 1) print_usage(argv[0]);
    argv = argv + optind;
    uint32_t mat_size = (uint32_t) atol(argv[0]);
    //
    // Allocate Buffers in Host
    int nBytes = sizeof(float) * mat_size * mat_size;
    float *hMat_A, *hMat_B, *hMat_C, *hMat_G, *hMat_S;
    if ((hMat_A = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_B = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_C = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_G = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    if ((hMat_S = (float*)malloc(nBytes)) == NULL) exit(EXIT_FAILURE);
    //
    // Generate Random Matrix
    time_t t;
    srand((unsigned int)time(&t));
    for (uint32_t i = 0; i < mat_size * mat_size; i++)
    {
        hMat_A[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
        hMat_B[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
    }
    //
    // Do each Calculation
    if (do_cpu) CPU_Main(hMat_A, hMat_B, hMat_C, mat_size);
    if (do_gpu_global) GPU_G_Main(hMat_A, hMat_B, hMat_G, mat_size);
    if (do_gpu_shared) GPU_S_Main(hMat_A, hMat_B, hMat_S, mat_size);
    //
    // Do Verification if possible
    if (do_cpu && do_gpu_global) Compare(hMat_C, hMat_G, mat_size, (char*)"GPU Global");
    if (do_cpu && do_gpu_shared) Compare(hMat_C, hMat_S, mat_size, (char*)"GPU Shared");
    //
    // Finish
    if (hMat_A) free(hMat_A);
    if (hMat_B) free(hMat_B);
    if (hMat_C) free(hMat_C);
    if (hMat_G) free(hMat_G);
    if (hMat_S) free(hMat_S);
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
