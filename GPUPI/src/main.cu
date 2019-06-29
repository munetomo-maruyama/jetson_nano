//====================================================
// Calculate any digit of Pi using BBP formula
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.04.28 M.Munetomo
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
#include "pi_cpu.h"
#include "pi_gpu.h"

#define DIGIT_SHOW 100

//-----------------
// Print Usage
//-----------------
void print_usage(char *cmd)
{
    printf("[Usage] %s [-c] [-g] digit\n", cmd);
    printf("    -c : Invoke CPU\n"); 
    printf("    -g : Invoke GPU\n"); 
    printf("    digit : Specify digit counts\n");
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
    if (argc < 1) print_usage(argv[0]);
    argv = argv + optind;
    
    int64_t digit = (int64_t) atoll(argv[0]);
    int64_t digit_show_bgn = (digit < DIGIT_SHOW)? 0 : digit -DIGIT_SHOW; // digit to be shown
    digit = ((digit + DIGIT_STEP - 1) / DIGIT_STEP) * DIGIT_STEP; // digit to be calculated
    
    //---------------------------------------------------
    // CPU Calculation
    //---------------------------------------------------
    if (do_cpu)
    {
        //
        // Start Message
        printf("--------[CPU] %lld digits Calculation...(showing specified last %d digits)\n", (long long int)digit, DIGIT_SHOW);
        //
        // Allocate Result Buffer
        char *cpu_result = NULL;
        cpu_result = (char*) malloc((digit + DIGIT_STEP) * sizeof(char));
        if (cpu_result == NULL) return(EXIT_FAILURE);
        memset(cpu_result, 0, digit + DIGIT_STEP);
        //
        // Calculate Pi by BBP formula
        double iStart = CPU_Second();
        int64_t d;
        for (d = 0; d < digit; d = d + DIGIT_STEP)
        {
            CPU_Calc_Pi(d, cpu_result);
        }
        double iElaps = CPU_Second() - iStart;
        //
        // Display Result
        for (d = digit_show_bgn; d < digit_show_bgn + DIGIT_SHOW; d++)
        {
            printf("%c", cpu_result[d]);
        }
        printf("\n");
        printf("Time elapsed %lf sec\n", iElaps);
        //
        // Finish
        free(cpu_result);
    }
    
    //---------------------------------------------------
    // GPU Calculation
    //---------------------------------------------------
    if (do_gpu)
    {
        //
        // Start Message
        printf("--------[GPU] %lld digits Calculation...(showing specified last %d digits)\n", (long long int)digit, DIGIT_SHOW);
        //
        // Allocate Result Buffer
        char *h_gpu_result = NULL;
        h_gpu_result = (char*) malloc((digit + DIGIT_STEP) * sizeof(char));
        if (h_gpu_result == NULL) return (EXIT_FAILURE);
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
        free(h_gpu_result);
    }
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
