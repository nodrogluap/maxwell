#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

// For warp level CUDA ops like shuffle down
#define FULL_MASK 0xffffffff

// Minimum hardware specs that we will support at runtime (Compute Capability 2.0)
#define CUDA_WARP_WIDTH 32
#if defined(_DEBUG_REG)
#define CUDA_THREADBLOCK_MAX_THREADS 256
#else
#define CUDA_THREADBLOCK_MAX_THREADS 1024
#endif
#define CUDA_THREADBLOCK_MAX_L1CACHE 48000
// It's actually most efficient computationally to send all the queries to the CUDA kernels en masse if you have a bunch of short ones, rather than sending them one at a time.
// Set a cap on the number to send at once since we need to know the limit ahead of time for task parcelling. 
#define CUDA_CONSTANT_MEMORY_SIZE 66068

// Convenience macro used when calculating number of blocks required for processing a given anmount of data
#define DIV_ROUNDUP(numerator, denominator) (((numerator) + (denominator) - 1)/(denominator))

#define CUERR(MSG) { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << MSG << ")" << std::endl; exit((int) err);}}

cudaEvent_t start, stop;
float __timer_milliseconds = 0;

#define START_TIMER(MSG) { cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    std::cerr << MSG; }

#define END_TIMER(MSG) {        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&__timer_milliseconds, start, stop); \
        std::cerr << __timer_milliseconds << "ms" << MSG << std::endl;}


#endif
