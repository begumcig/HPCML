#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//--- Returns a Random Float. ---//
float fRand(int range)
{
    float f = ((float)rand() / RAND_MAX) * range;
    return f;
}

//--- Dot Product Function. ---//
float dpunroll(long N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j = 0; j < N; j += 4)
        R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];
    return R;
}

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        printf("Too many or not enough arguments!");
        return 0;
    }

    printf("\n---------- NOTE ---------- \nPrinting the result at every loop to avoid optimization.\nPlease do not remove.");
    printf("--------------------------\n");

    long N = atol(argv[1]);
    long iter = atol(argv[2]);
    //printf("N is %lu, loop count is %lu.\n", N, iter);

    srand((unsigned)time(NULL));
    int range = 10;

    struct timespec start, end;
    float result, timeSum, timeAvg;

    //--- Declaration & Initialization of Arrays. ---//
    float *pA = (float *)malloc(sizeof(float) * N);
    float *pB = (float *)malloc(sizeof(float) * N);
    float *timeElapsArr = (float *)malloc(sizeof(float) * iter);

    for (int i = 0; i < N; i++)
    {
        pA[i] = fRand(range);
        pB[i] = fRand(range);
        //printf("Created two random numbers %.3f and %.3f respectively.\n", pA[i], pB[i]);
    }

    //--- Benchmark Calculation. ---///
    for (int i = 0; i < iter; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        result = dpunroll(N, pA, pB);
        printf("result: %.3f.\n", result);
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculated in micro seconds, then divided by 10^6 to get seconds.
        double timeUsed = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec / 1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec / 1000)) / 1000000;
        float flops = (float)(N * 2) / (timeUsed * 1000000000);
        float gbs = (N * 2 * sizeof(float)) / (timeUsed * 1000000000);
        timeElapsArr[i] = timeUsed;
        printf("N: %lu <R>: %.3f <T>: %.6f sec B: %.3f GB/sec F: %.3f GFLOP/sec \n", N, result, timeUsed, gbs, flops);
    
    }

    //--- Time Average. ---///
    timeSum = 0.0;
    for (int i = (iter / 2); i < iter; i++)
    {
        timeSum += timeElapsArr[i];
    }

    timeAvg = timeSum / (float)(iter / 2);
    //printf("Average time: %.6f\n", timeAvg);

    //--- Flops and Memory ---//

    // 2 float operations for loop size of N.
    // time is calculated in seconds so no further translation.
    float flops = (float)((N / 4) * 8) / (timeAvg * 1000000000);

    // 2 memory operations (pA and pB) for loop size of N.
    // the size of floats in bytes, so divided by 10^9 to reach gb/s.
    float gbs = ((N/4) * 8 * sizeof(float)) / (timeAvg * 1000000000);
    //printf("flops: %.6f\n, gbs: %.6f\n ", flops, gbs);

    printf("N: %lu <T>: %.6f sec B: %.3f GB/sec F: %.3f GFLOP/sec \n", N, timeAvg, gbs, flops);

    return 0;
}