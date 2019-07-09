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

#include "common.h"
#include "do_cpu.h"
#include "do_gpu.h"

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
    //
    // CPU Calculation
    if (do_cpu) CPU_Main(digit);
    //   
    // GPU Calculation
    if (do_gpu) GPU_Main(digit);
    //
    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
