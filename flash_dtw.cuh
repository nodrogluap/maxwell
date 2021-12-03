/* FLASH DTW

   Fast Locally Aggregated Subalignment Heuristic for Dynamic Time Warping

   Paul Gordon, 2019
   Edited: Steven Hepburn, 2019
   Edited: Matthew Wiens, 2019


   The idea here is to find points in the middle of the local warp path that must be passed through, which
   lets us reduce the cost matrix size a lot since we can compute a separate, smaller cost matrix for each of the 
   between-anchor sections and then stitch the warp paths together (hence stitch_dtw function below). 
   If you had a semi-global (a.k.a. streaming) DTW implementation to find the
   location of the query in a large subject based on collinearity (within a band), you'd already know the anchor points,
   i.e. the collinear blocks. This saves us effort compared to sparseDTW or other global warp path efficiency methods. 
   In a way you could consider this a "seed-and-extend" approach to DTW. 
   We do this most efficiently by calculating tiny DTW blocks on all the subject in parallel on the GPU, then only fully align the 
   bits that are non-randomly less distant than other non-collinear blocks, using a Mann-Whitney test of tiny DTW distances ranked 
   for co-linear vs non., and FDR correction if requested. 

   This header file contains a bunch of CUDA and host functions, but only those with a prototype at the top of this file are for calling externally. 

   TODO:
	- somehow passing back matching blocks = 0 with command line './flash_test ../myFiles/k_segmentation_v1/ch347_read8418_segmented_values.raw.txt ../myFiles/k_segmentation_v1/ch347_read8418_noheader.raw.txt -n 3 -v -c -l 10 -g 2 -w 2', fix this, seems to only occur when using '-g 2'
	- test soft_dtw and hard_dtw (may not be working, time is same for all sizes ???)
	- enable reading of fast5 files
	- review and enable half, short, and double data types
	- review and enable MINIDTW_STRIDE != 1
	- every kernel needs unit testing
	- print out info only when verbose is enabled (even in flash_dtw.cuh side functions)
*/

#ifndef __FLASH_DTW_H
#define __FLASH_DTW_H

/* NVIDIA GPUs provide 64KB of constant cache, which is accessed as somewhere between 2 and 8KB of dedicated L1 cache on each multiprocessor (a.k.a. SM or SMX).
   Let's devote almost all of it to the query to give us the most flexibility possible in terms of searching long queries. For shorts (default) that means ~32K is max query length. */
// Unlike many CPU-based lower-bound optimized implementations of streaming DTW, we need to store all the query in limited device memory, so we really care about how
// we will store the query (and subject) values as bigger datatypes like floats limit the size of DTW we can compute compared to char or short.
#ifdef QTYPE_short
#define QTYPE short
#endif
// Should be big enough to hold dynamic range of QTYPE values*MAX_DP_SAMPLES
#ifdef QTYPE_ACC_short
#define QTYPE_ACC short
#define DTW_MAX SHRT_MAX
#define dtwmin(a, b) (min(a, b))
#endif

#ifdef QTYPE_float
#define QTYPE float
#endif

#ifdef QTYPE_ACC_float
#define QTYPE_ACC float
#define DTW_MAX FLT_MAX
#define dtwmin(a, b) (min(a, b))
#endif

#ifdef QTYPE_double
#define QTYPE double
#endif

#ifdef QTYPE_ACC_double
#define QTYPE_ACC double
#define dtwmin(a, b) (min(a, b))
#define DTW_MAX DBL_MAX
#endif

#ifdef QTYPE_half
#include <cuda_fp16.h>
#define QTYPE __half
#endif

#ifdef QTYPE_ACC_half
#define QTYPE_ACC __half
#define DTW_MAX 65504.f
#define dtwmin(a, b) (__hgt(a, b) ? a : b)
#endif

#include "algo_datatypes.h"

// Here are prototypes for the methods that the caller cares about.
__host__ void load_subject(QTYPE *subject, QTYPE *subject_std, long subject_length, int use_std, cudaStream_t stream);
__host__ int flash_dtw(QTYPE *query, int query_length, char* query_name, float max_warp_proportion, float max_pvalue, float max_qvalue, int max_ranks, int normalization_mode, match_record **results, int *num_results, int minidtw_size, int minidtw_warp, bool record_anchors, int use_fast_anchor_calc, int use_std, int use_hard_dtw, cudaStream_t stream);
__inline__ __host__ void soft_dtw_wrap(const dim3 griddim_dtw, const int threadblock_size_dtw, cudaStream_t stream, const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, const int minidtw_size, const int minidtw_warp, const int use_std);

// Here are prototypes for the methods we want to unit test
template <class T>
__host__ void get_znorm_stats(T *data, long data_length, float *mean, float *stddev, T *min, T *max, cudaStream_t stream=0);

// __global__ void mean_min_max_float(float *data, int data_length, float *threadblock_means, QTYPE *threadblock_mins, QTYPE *threadblock_maxs);
// __global__ void mean_min_max(QTYPE *data, int data_length, float *threadblock_means, QTYPE *threadblock_mins, QTYPE *threadblock_maxs);
template <class T>
__global__ void variance(T *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances);
__global__ void variance_float(float *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances);
template <class T>
__global__ void query_znorm(int offset, size_t length, float *data_mean, float *data_stddev, T *min, T *max, int normalization_mode, cudaStream_t stream=0);
__global__ void welford_query_znorm(int offset, int length, float* mean, float* ssq, long total_values_znormalized, cudaStream_t stream=0);
__global__ void thorough_calc_anchor_candidates_colinear_pvals(const long long *query_adjacent_candidates, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, const int expected_num_pvals, const float max_warp_proportion, const float max_pval, const QTYPE_ACC *query_adjacent_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, const int minidtw_size, int* all_memberships, long long* all_subject_indices, int* lis_memberships, QTYPE_ACC* colinear_buff, QTYPE_ACC* non_colinear_buff, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff);
__global__ void fast_calc_anchor_candidates_colinear_pvals(const long long *query_adjacent_candidates, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, int *num_non_colinear_distances, const float max_warp_proportion, const float max_pval, const QTYPE_ACC *query_adjacent_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, const int minidtw_size, int* all_memberships, long long* all_subject_indices, int* lis_memberships, QTYPE_ACC* colinear_buff, QTYPE_ACC* non_colinear_buff, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff);

__global__ void load_non_colinear_distances(const QTYPE_ACC *query_adjacent_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, int *num_non_colinear_distances);

__global__ void fdr(float *input_pvals, int *input_pval_ranks, unsigned int input_pvals_length, float *output_qvals);

__global__ void hard_dtw_std(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances);
__global__ void hard_dtw(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, int minidtw_size);
__global__ void soft_dtw_std(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances);
__global__ void soft_dtw(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances);

// Here are prototypes for the inline methods we want to unit test
__inline__ __device__ float warpReduceSum(float val);
__inline__ __device__ int warpReduceMin(int val);
template <class T>
__inline__ __device__ void set_membership_results(T* membership, const int num_candidate_query_indices, long long* subject_indices, const int minidtw_size, const float max_warp_proportion, bool global_mem);
template <class T>
__inline__ __device__ void get_sorted_colinear_distances(QTYPE_ACC *sorted_colinear_distances, T* membership, QTYPE_ACC *distances, const int num_candidate_query_indices, int* partition_counter, int* local_partition, int num_members, int* num_sorted_colinear_distances, bool global_mem);
template <class T>
__inline__ __device__ void get_sorted_non_colinear_distances(QTYPE_ACC *sorted_non_colinear_distances, T* membership, int num_members, const QTYPE_ACC *query_adjacent_distances, int* num_sorted_non_colinear_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, bool thorough_calc, QTYPE_ACC *global_non_colinear_distances, int num_sorted_colinear_distances, bool global_mem);
__inline__ __device__ void calculate_pval(const float max_pval, QTYPE_ACC *sorted_colinear_distances, QTYPE_ACC *sorted_non_colinear_distances, int num_sorted_non_colinear_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int leftmost_anchor_query, int rightmost_anchor_query, long leftmost_anchor_subject, long rightmost_anchor_subject, int num_members, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff);
__host__ __inline__ int flash_dtw_setup(QTYPE* query, int query_length, int normalization_mode, cudaStream_t stream=0);
__inline__ __device__ void block_findMin(QTYPE *dtw_results, long long *idata, const int tid, const int block_dim, const long long strided_jobid, QTYPE_ACC &min, long long &index);

// Functions that are no longer in use
__host__ QTYPE* segment_signal(QTYPE *series, unsigned int *series_length, short expected_segment_length, float max_attenuation, int normalization_mode, cudaStream_t stream);
__global__ void merge_ranks(float *anchor_pvals, int num_pvals, int max_rank, int *anchor_pval_ranks, unsigned int *anchor_num_ranks);
__global__ void rank_results_blocks(float *input_pvals, int input_pvals_length, int *sorted_pval_ordinals, int *sorted_pval_ranks);
__global__ void get_min_distance_sorted_subject_sample(QTYPE_ACC *dtw_distances, int *min_idxs, int num_min_idxs, QTYPE_ACC *sorted_non_colinear_distances, int num_samples);
__host__ double stitch_dtw(std::vector< std::pair<int,long> > input_warp_anchors, float *query, float *subject, std::vector< std::pair<int,long> > &output_warp_path);
__host__ void traceback(double *matrix, int nx, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path);
__host__ double euclidean_dtw(float *x, int nx, float *y, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path);

#include <thrust/sort.h>
#include "stdio.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
// For INT_MAX
#include <climits>
// For FLT_MAX etc.
#include <cfloat>
// For erf function
#include <cmath>

#include <fstream>

// For warp level CUDA ops like shuffle down
// #define FULL_MASK 0xffffffff

// #define CUDA_WARP_WIDTH 32
// #if defined(_DEBUG_REG)
// #define CUDA_THREADBLOCK_MAX_THREADS 256
// #else
// #define CUDA_THREADBLOCK_MAX_THREADS 1024
// #endif
// #define CUDA_THREADBLOCK_MAX_L1CACHE 48000

#include "cuda_utils.h"
   
// MAX_DP_SAMPLES should not exceed 256. Otherwise, sums and squares accumulators used in the segmentation kernel could overflow in edge cases of extremely noisy, high dynamic range signal.
#define MAX_DP_SAMPLES 256

// How should queries be Z-normalized?
#define PERCENTILE_NORM 4 // Scale so that the 5th percentile and 95th percentile value are the same in the query and subject
#define GLOBAL_ZNORM 3 // Across all queries that are loaded together
#define ONLINE_ZNORM 2 // Welford's method on each individual query
#define LOCAL_ZNORM 1  // Each query separately
#define NO_ZNORM 0     // Use as-is (i.e. it came in z-normalized)

// #define CUDA_CONSTANT_MEMORY_SIZE 66068
#define MAX_NUM_QUERIES 512
__device__ volatile int Cnum_nonzero_series = 0;
__device__ volatile short Cnonzero_series_lengths[MAX_NUM_QUERIES];
#define MAX_QUERY_SIZE CUDA_CONSTANT_MEMORY_SIZE/sizeof(QTYPE)-sizeof(short)*MAX_NUM_QUERIES-sizeof(int)-sizeof(long)-sizeof(float)*2
__device__ QTYPE Gquery[MAX_QUERY_SIZE];
__device__ QTYPE Gquery_std[MAX_QUERY_SIZE];
__device__ volatile int Gquery_length;

// How many positions to skip between anchors
#ifndef MINIDTW_STRIDE
#define MINIDTW_STRIDE 4 // can only be one of {1, 2, 4, 8}
#endif

// It turns out that it's crazy expensive computationally to just keep the best N match results (e.g. something reasonable like CUDA_THREADBLOCK_MAX_THREADS), because 
// it requires tonnes of super-inefficient atomic operations and threadfences to coordinate the parallel threads across
// the device in updating the results array. Instead, allocate memory for and record up to MAX_PVAL_MATCHES_KEPT matches and only count the number beyond that.
// CUDA_THREADBLOCK_MAX_THREADS*CUDA_THREADBLOCK_MAX_THREADS = ~One million and means that the FDR correction for something as common as the Alu repeat in a 
// human genome will be a reasonable estimate, and we only need to do one reduction step to determine match ranks (necessary for FDR calculation).
#ifndef MAX_PVAL_MATCHES_KEPT
#define MAX_PVAL_MATCHES_KEPT (CUDA_THREADBLOCK_MAX_THREADS*CUDA_THREADBLOCK_MAX_THREADS)
#endif

__device__ __constant__ long Tsubject_length = 0;
// And here is the underlying device global memory that actually stores the search subject
QTYPE *Dsubject = 0;
QTYPE *Dsubject_std = 0;
// We need to keep track of the stddev because we don't really Z-normalize (since this would require at least an fp16 for each subject value, which is silly if QTYPE is 8 bits)
// but rather adjust the query to have the same mean and std dev as the subject. 
__device__ __constant__ float Dsubject_mean = 0;
__device__ __constant__ float Dsubject_stddev = 0;
// Also record the dynamic range of the subject, in case someone is interested in comparing it to the dynamic range of the normalized queries later, "for training and quality assurance purposes".
__device__ float Dsubject_min = 0;
__device__ float Dsubject_max = 0;

template<class T>
__host__
void load_and_normalize_queries(T *query_values, size_t query_length, int normalization_mode, float* welford_mean, float* welford_ssq, long total_values_znormalized, cudaStream_t stream=0){
	
	
	cudaMemcpyToSymbol(Gquery, query_values, sizeof(T)*query_length);                            CUERR("Copying Gquery from CPU to GPU emory")
	cudaMemcpyToSymbolAsync(::Gquery_length, &query_length, sizeof(int), 0, cudaMemcpyHostToDevice, 0);		CUERR("Copying query's length from CPU to GPU constant memory")
	
	float *mean, *stddev;
    T *min, *max;
    cudaMalloc(&mean, sizeof(float)); CUERR("Allocating GPU memory for query mean");
    cudaMalloc(&stddev, sizeof(float)); CUERR("Allocating GPU memory for query std dev");
    cudaMalloc(&min, sizeof(T)); CUERR("Allocating GPU memory for query min");
    cudaMalloc(&max, sizeof(T)); CUERR("Allocating GPU memory for query max");
	cudaStreamSynchronize(stream);
	// Next two line used so the query location can be passed around like any other memory pointer to the stats function.
	T *query_addr = 0;
	cudaGetSymbolAddress((void **) &query_addr, Gquery);    CUERR("Getting memory address of device constant for query");
	if(normalization_mode == GLOBAL_ZNORM){
		dim3 norm_grid(DIV_ROUNDUP(query_length,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		get_znorm_stats<T>(query_addr, query_length, mean, stddev, min, max, stream);     CUERR("Setting global query stats");
		cudaStreamSynchronize(stream);				CUERR("Synchronizing stream after query znorm stats");
		
		// float *host_mean, *host_stddev;
		// T *host_min, *host_max;
		
		// host_mean = (float*)malloc(sizeof(float));
		// host_stddev = (float*)malloc(sizeof(float));
		// host_min = (T*)malloc(sizeof(T));
		// host_max = (T*)malloc(sizeof(T));
		
		// cudaMemcpy(host_mean, mean, sizeof(float), cudaMemcpyDeviceToHost);
		// cudaMemcpy(host_stddev, stddev, sizeof(float), cudaMemcpyDeviceToHost);
		// cudaMemcpy(host_min, min, sizeof(T), cudaMemcpyDeviceToHost);
		// cudaMemcpy(host_max, max, sizeof(T), cudaMemcpyDeviceToHost);
		
		// std::cerr << "mean: " << *host_mean << ", stddev: " << *host_stddev << ", min: " << *host_min << ", max: " << *host_max << std::endl;
		
		query_znorm<T><<<norm_grid, CUDA_THREADBLOCK_MAX_THREADS, 0, stream>>>(0, query_length, mean, stddev, min, max, normalization_mode);              CUERR("Applying global query normalization in segment_and_load_queries GLOBAL_ZNORM");
		cudaStreamSynchronize(stream);				CUERR("Synchronizing stream after query znorm");
	}
	else if(normalization_mode == LOCAL_ZNORM){
		// Not really worth parallelizing this outer loop as the size of the query is very constrained for now
		dim3 grid(DIV_ROUNDUP(query_length,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		get_znorm_stats<T>(query_addr, query_length, mean, stddev, min, max, stream);     CUERR("Setting global query stats");
		query_znorm<T><<<grid,CUDA_THREADBLOCK_MAX_THREADS,0,stream>>>(0, query_length, mean, stddev, min, max, normalization_mode);	CUERR("Applying local query normalization");
	}
	else if(normalization_mode == PERCENTILE_NORM){
	}
	else if(normalization_mode == ONLINE_ZNORM) {
		float* tmp_mean, *tmp_ssq;
		cudaMalloc(&tmp_mean, sizeof(float)); CUERR("Allocating GPU memory for temp welford mean");
		cudaMalloc(&tmp_ssq, sizeof(float)); CUERR("Allocating GPU memory for temp welford ssq");
		cudaMemcpy(tmp_mean, welford_mean, sizeof(float), cudaMemcpyHostToDevice);	CUERR("Copying welford mean from host to device");
		cudaMemcpy(tmp_ssq, welford_ssq, sizeof(float), cudaMemcpyHostToDevice);		CUERR("Copying welford ssq from host to device");
		welford_query_znorm<<<1, CUDA_THREADBLOCK_MAX_THREADS, 0, stream>>>(0, query_length, tmp_mean, tmp_ssq, total_values_znormalized); 
		cudaMemcpy(welford_mean, tmp_mean, sizeof(float), cudaMemcpyDeviceToHost);	CUERR("Copying welford mean from device to host");
		cudaMemcpy(welford_ssq, tmp_ssq, sizeof(float), cudaMemcpyDeviceToHost);		CUERR("Copying welford ssq from device to host");
		cudaFree(tmp_mean);	CUERR("Freeing temp welford mean");
		cudaFree(tmp_ssq);	CUERR("Freeing temp welford ssq");
		cudaStreamSynchronize(stream);				CUERR("Synchronizing stream after query znorm");
	}
	else{
		dim3 norm_grid(DIV_ROUNDUP(query_length,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		// Next two line used so the query location can be passed around like any other memory pointer to the stats function.
		query_znorm<T><<<norm_grid, CUDA_THREADBLOCK_MAX_THREADS, 0, stream>>>(0, query_length, mean, stddev, min, max, normalization_mode);              CUERR("Applying global query normalization in segment_and_load_queries no GLOBAL_ZNORM");
		cudaStreamSynchronize(stream);				CUERR("Synchronizing stream after query znorm");
	}
	cudaFree(mean);							CUERR("Freeing query mean");
	cudaFree(stddev);						CUERR("Freeing query stddev");
	cudaFree(min);							CUERR("Freeing query min");
	cudaFree(max);							CUERR("Freeing query max");
}

// The traceback and full dtw methods use float arguments and double accumulators/results regardless of the QTYPE, because we are Z-normalizing the query.
__host__
void traceback(double *matrix, int nx, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path){
    int i = nx-1;
    int j = ny-1;
    path.push_back(std::make_pair(xoffset+i,yoffset+j)); // end anchor
    while ((i > 0) && (j > 0)){
        // 3 conditions below are equivalent to min((matrix[i-1, j-1], matrix[i, j-1], matrix[i-1, j]))
        if (matrix[i-1+(j-1)*nx] <= matrix[i+(j-1)*nx] && matrix[i-1+(j-1)*nx] <= matrix[(i-1)+j*nx]){ // diagonal move is preferred (i.e. White-Neely step scoring)
            --i;
            --j;
        }
        else if (matrix[i+(j-1)*nx] <= matrix[i-1+(j-1)*nx] && matrix[i+(j-1)*nx] <= matrix[(i-1)+j*nx])
            --j;
        else
            --i;
        path.push_back(std::make_pair(i+xoffset,j+yoffset));
        //std::cout << matrix[i+j*nx] << "/";
    }
    while(i > 0){
        path.push_back(std::make_pair((--i)+xoffset,yoffset));
    }
    while(j > 0){
        path.push_back(std::make_pair(xoffset,(--j)+yoffset));
    }
    // flip the path back to ascending order
    std::reverse(path.begin(), path.end());
    //std::cout << std::endl;
}

__host__
double euclidean_dtw(float *x, int nx, float *y, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path){
    QTYPE max = std::numeric_limits<QTYPE>::max();
    
    double *accumulated_cost_matrix = (double *) malloc(sizeof(double)*nx*ny);
    if(accumulated_cost_matrix == 0){
      std::cerr << "Could not allocate cost matrix for DTW of size (" << nx << "," << ny << "), aborting" << std::endl;
      exit(1);
    }
    for(int i = 1; i < nx; ++i){
      accumulated_cost_matrix[i] = (double) max;
    }
    for(int i = 1; i < ny; ++i){
      accumulated_cost_matrix[i*nx] = (double) max;
    }
    accumulated_cost_matrix[0] = abs(x[0]-y[0]);
    for(int i = 1; i < nx; ++i){
        for(int j = 1; j < ny; ++j){
            // 3 conditions below are equivalent to min((matrix[i-1, j-1], matrix[i, j-1], matrix[i-1, j]))
            if(accumulated_cost_matrix[i-1+(j-1)*nx] <= accumulated_cost_matrix[i+(j-1)*nx] && accumulated_cost_matrix[i-1+(j-1)*nx] <= accumulated_cost_matrix[i-1+j*nx]){
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i-1+(j-1)*nx];
            }
            else if(accumulated_cost_matrix[i+(j-1)*nx] <= accumulated_cost_matrix[i-1+(j-1)*nx] && accumulated_cost_matrix[i+(j-1)*nx] <= accumulated_cost_matrix[i-1+j*nx]){
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i+(j-1)*nx];
            }
            else{
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i-1+j*nx];
            }
        }
    }
    traceback(accumulated_cost_matrix, nx, ny, xoffset, yoffset, path);
    double cost = accumulated_cost_matrix[nx*ny-1];
    free(accumulated_cost_matrix);
    return cost;
}

/* returned path vector is pairs <query_position,subject_position> */
__host__
double stitch_dtw(std::vector< std::pair<int,long> > input_warp_anchors, float *query, float *subject, std::vector< std::pair<int,long> > &output_warp_path){
   //TODO: align head and tail outside the anchors?
   double total_cost = 0;
   for(int i = 0; i < input_warp_anchors.size()-1; ++i){
       std::vector< std::pair<int,long> > local_path;
       local_path.reserve((input_warp_anchors[i+1].first-input_warp_anchors[i].first)*1.5);
       //std::cout << "Stitching query (" << input_warp_anchors[i].first << "," << input_warp_anchors[i+1].first << ") with subject (" << input_warp_anchors[i].second << "," << input_warp_anchors[i+1].second-1 << "), cost ";
       double cost = euclidean_dtw(&query[input_warp_anchors[i].first], input_warp_anchors[i+1].first-input_warp_anchors[i].first, 
                     &subject[input_warp_anchors[i].second], input_warp_anchors[i+1].second-input_warp_anchors[i].second, 
                     input_warp_anchors[i].first, input_warp_anchors[i].second, local_path);
       //std::cout << cost << std::endl;
       total_cost += cost;
       output_warp_path.insert(output_warp_path.end(), local_path.begin(), local_path.end());
   } 
   return total_cost;
}

// Adds all values in a warp and returns the result
// val - the value in the current block of the warp to be added with the rest
// returns the final added value
__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(FULL_MASK, val, offset);
    //val += __shfl_down(val, offset);
  return val;
}

// Min/ max reduction code editied from: https://ernie55ernie.github.io/parallel%20programming/2018/03/17/cuda-maximum-value-with-parallel-reduction.html
template <class T>
__global__
void mean_min_max_float(float *data, int data_length, float *threadblock_means, T *threadblock_mins, T *threadblock_maxs){

	__shared__ float warp_sums[CUDA_WARP_WIDTH];
  
	extern __shared__ T cache[];
	// extern __shared__ T min_cache[];
  
	T tmp_max = threadblock_maxs[0];
	T tmp_min = threadblock_mins[0];
  
	int pos = blockIdx.x*blockDim.x+threadIdx.x;
	float warp_sum = 0;
	if(pos < data_length){ // Coalesced global mem reads
		warp_sum = data[pos]; 
	}
	__syncwarp();

	// Reduce the warp
	warp_sum = warpReduceSum(warp_sum);
	if(threadIdx.x%CUDA_WARP_WIDTH == 0){
		warp_sums[threadIdx.x/CUDA_WARP_WIDTH] = warp_sum;
	}
	__syncwarp();
	__syncthreads();

	int warp_limit = DIV_ROUNDUP(blockDim.x,CUDA_WARP_WIDTH);
	// Reduce the whole threadblock
	if(threadIdx.x < CUDA_WARP_WIDTH){
		warp_sum = threadIdx.x < warp_limit ? warp_sums[threadIdx.x] : 0;
		__syncwarp();
		warp_sum = warpReduceSum(warp_sum);
	}
  
	int cacheIndex = threadIdx.x;
  
	while(pos < blockDim.x){
		if(threadblock_maxs[pos] > tmp_max){
			tmp_max = threadblock_maxs[pos];
		}
		if(threadblock_mins[pos] < tmp_min){
			tmp_min = threadblock_mins[pos];
		}
		pos += blockDim.x * gridDim.x; 
	}
  
	cache[cacheIndex] = tmp_max;
	cache[cacheIndex + blockDim.x] = tmp_min;
  
	__syncthreads();
  
	// printf("cache[%i]: %f, cache[%i]: %f\n", cacheIndex, cache[cacheIndex], cacheIndex + blockDim.x, cache[cacheIndex + blockDim.x]);
  
	int ib = blockDim.x / 2;
	// printf("ib: %i\n", ib);
	while (ib != 0) {
		// printf("cache[%i]: %f, cache[%i + %i]: %f\n", cacheIndex, cache[cacheIndex], cacheIndex, ib, cache[cacheIndex + ib]);
		if(cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex]){
			cache[cacheIndex] = cache[cacheIndex + ib]; 
		}
		// printf("cache[%i + %i]: %f, cache[%i+ %i + %i]: %f\n", blockDim.x, cacheIndex, cache[blockDim.x + cacheIndex], blockDim.x, cacheIndex, ib, cache[blockDim.x + cacheIndex + ib]);
		if(cacheIndex < ib && cache[blockDim.x + cacheIndex + ib] < cache[blockDim.x + cacheIndex]){
			cache[blockDim.x + cacheIndex] = cache[blockDim.x + cacheIndex + ib]; 
		}
		__syncthreads();

		ib /= 2;
	}
	
	if(cacheIndex == 0){
		threadblock_maxs[0] = cache[0];
		threadblock_mins[0] = cache[blockDim.x];
	}
	// Assign min/ max value to the front
	// threadblock_mins[0] = threadblock_mins[pos] < threadblock_mins[0] ? threadblock_mins[pos] : threadblock_mins[0];
	// threadblock_maxs[0] = threadblock_maxs[pos] > threadblock_maxs[0] ? threadblock_maxs[pos] : threadblock_maxs[0];
	// if(threadIdx.x == 0){
		// for(int i = 0; i < blockDim.x; i++){
			// threadblock_mins[0] = threadblock_mins[i] < threadblock_mins[0] ? threadblock_mins[i] : threadblock_mins[0];
			// threadblock_maxs[0] = threadblock_maxs[i] > threadblock_maxs[0] ? threadblock_maxs[i] : threadblock_maxs[0];
		// }
	// }
	// Assign to global memory for later reduction across the whole data array
	if(! threadIdx.x){
		// Special condition in denominator for the last, potentially incomplete block
		threadblock_means[blockIdx.x] = warp_sum/((blockIdx.x+1)*blockDim.x > data_length ? data_length-blockIdx.x*blockDim.x : blockDim.x);
	}
}

template <class T>
__global__
void
mean_min_max(T *data, int data_length, float *threadblock_means, T *threadblock_mins, T *threadblock_maxs){

  __shared__ T threadblock_data[CUDA_THREADBLOCK_MAX_THREADS];
  __shared__ float warp_sums[CUDA_WARP_WIDTH];
  __shared__ T warp_mins[CUDA_WARP_WIDTH];
  __shared__ T warp_maxs[CUDA_WARP_WIDTH];

  if(threadIdx.x < CUDA_WARP_WIDTH){
    warp_mins[threadIdx.x] = sizeof(T) == 1 ? UCHAR_MAX : (sizeof(T) == 2 ? SHRT_MAX : FLT_MAX);
    warp_maxs[threadIdx.x] = sizeof(T) == 1 ? 0 : (sizeof(T) == 2 ? SHRT_MIN : FLT_MIN);
  }
  __syncwarp();
  __syncthreads();

  // This same method is used to calculate stats on both data in global memory (e.g. subject), and device constants (e.g. query).
  // To allow both of these to be processed correctly, we can indicate that we're using the query by passing in.
  if(data == 0){
    data = (T*)&Gquery[0];
  }

  int pos = blockIdx.x*blockDim.x+threadIdx.x;
  float warp_sum = 0;
  if(pos < data_length){ // Coalesced global mem reads
  // printf("data[%i]: %lf\n", pos, data[threadIdx.x]);
    threadblock_data[threadIdx.x] = data[pos];
    warp_sum = (double) threadblock_data[threadIdx.x]; // __shfl*() only works with int or float, the latter is the safest bet to retain true data values regardless of QTYPE
  }
  __syncwarp();

  // Reduce the warp
  warp_sum = warpReduceSum(warp_sum);
  if(threadIdx.x%CUDA_WARP_WIDTH == 0){
    T warp_max = sizeof(T) == 1 ? 0 : (sizeof(T) == 2 ? SHRT_MIN : FLT_MIN);
    T warp_min = sizeof(T) == 1 ? UCHAR_MAX : (sizeof(T) == 2 ? SHRT_MAX : FLT_MAX);

    for(int i = 0; i < CUDA_WARP_WIDTH && threadIdx.x+i < (data_length-blockIdx.x*blockDim.x); i++){
      if(threadblock_data[threadIdx.x+i] < warp_min) warp_min = threadblock_data[threadIdx.x+i];
      if(threadblock_data[threadIdx.x+i] > warp_max) warp_max = threadblock_data[threadIdx.x+i];
    }
    warp_mins[threadIdx.x/CUDA_WARP_WIDTH] = warp_min;
    warp_maxs[threadIdx.x/CUDA_WARP_WIDTH] = warp_max;
    warp_sums[threadIdx.x/CUDA_WARP_WIDTH] = warp_sum;
    //printf("Block %d, Warp (%d) sum: %f\n", blockIdx.x, warp_sums[threadIdx.x/CUDA_WARP_WIDTH]);
  }
  __syncwarp();
  __syncthreads();

  int warp_limit = DIV_ROUNDUP(blockDim.x,CUDA_WARP_WIDTH);
  // Reduce the whole threadblock
  if(threadIdx.x < CUDA_WARP_WIDTH){
	  //printf("threadIdx %i, Warp warp_limit (%i) warp_sums[threadIdx.x]: %f\n", threadIdx.x, threadIdx.x, warp_sums[threadIdx.x]);
    warp_sum = threadIdx.x < warp_limit ? warp_sums[threadIdx.x] : 0;
    for(int swath = 1; threadIdx.x+swath < warp_limit; swath *= 2){
      if(threadIdx.x%(swath*2) == 0){
        if(warp_mins[threadIdx.x] > warp_mins[threadIdx.x+swath]) warp_mins[threadIdx.x] = warp_mins[threadIdx.x+swath]; 
        if(warp_maxs[threadIdx.x] < warp_maxs[threadIdx.x+swath]) warp_maxs[threadIdx.x] = warp_maxs[threadIdx.x+swath]; 
      }
    }
    __syncwarp();
    // printf("Block %d, Warp reduce (%d) sum: %f\n", blockIdx.x, threadIdx.x, warp_sum);
    warp_sum = warpReduceSum(warp_sum);
  }

  // Assign to global memory for later reduction across the whole data array
  if(! threadIdx.x){
    // Special condition in denominator for the last, potentially incomplete block
	// printf("%f/((%i+1)*%i, %i, %i-%i*%i, %i", warp_sum, blockIdx.x, blockDim.x, data_length, data_length, blockIdx.x, blockDim.x, blockDim.x);
    threadblock_means[blockIdx.x] = warp_sum/((blockIdx.x+1)*blockDim.x > data_length ? data_length-blockIdx.x*blockDim.x : blockDim.x);
    threadblock_mins[blockIdx.x] = warp_mins[0];
    threadblock_maxs[blockIdx.x] = warp_maxs[0];
  }
}

template <class T>
__global__
void
variance(T *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances){
  int pos = blockIdx.x*blockDim.x+threadIdx.x;

  // This same method is used to calculate stats on both data in global memory (e.g. subject), and device constants (e.g. query).
  // To allow both of these to be processed correctly, we can indicate that we're using the query by passing in
  if(data == 0){
    data = (T*)&Gquery[0];
  }

  float warp_var = 0;
  if(pos < data_length){ // Coalesced global mem reads
    warp_var = data[pos]-data_mean[0];
    warp_var *= warp_var;
  }

  // Reduce the warp
  __syncwarp();
  __shared__ T warp_variances[CUDA_WARP_WIDTH];
  warp_var = warpReduceSum(warp_var);
  if(threadIdx.x%CUDA_WARP_WIDTH == 0){
    warp_variances[threadIdx.x/CUDA_WARP_WIDTH] = warp_var;
  }
  __syncwarp();
  __syncthreads();

  // Reduce the whole threadblock and assign result to global memory for later reduction across the whole data array
  int warp_limit = DIV_ROUNDUP(blockDim.x,CUDA_WARP_WIDTH);
  if(threadIdx.x < CUDA_WARP_WIDTH){
    warp_var = threadIdx.x < warp_limit ? warp_variances[threadIdx.x] : 0;
    // *NOTA BENE*: if CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH > CUDA_WARP_WIDTH this reduce will be incomplete!! 
    // As of 2019 this is not the case for any CUDA device (typically 1028/32, but can be 512/32 on older GPU cards).
    __syncwarp();
    warp_var = warpReduceSum(warp_var);
  }
  if(! threadIdx.x){
      // If this is the final reduction calculate the std dev from the variance
      if(gridDim.x == 1) 
        threadblock_variances[0] = (float) sqrt(warp_var/orig_data_length);	// Whole population not the sample
      else
        threadblock_variances[blockIdx.x] = warp_var;
  }
}

// For reducing step, variance accumulation is double regardless of QTYPE
__global__
void
variance_float(float *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances){
  int pos = blockIdx.x*blockDim.x+threadIdx.x;

  __shared__ double dwarp_variances[CUDA_WARP_WIDTH];
  float warp_var = 0;
  if(pos < data_length){ // Coalesced global mem reads
    warp_var = data[pos];
  }

  // Reduce the warp
  __syncwarp();
  warp_var = warpReduceSum(warp_var);
  if(threadIdx.x%CUDA_WARP_WIDTH == 0){
    dwarp_variances[threadIdx.x/CUDA_WARP_WIDTH] = warp_var;
  }
  __syncwarp();
  __syncthreads();

  // Reduce the whole threadblock and assign result to global memory for later reduction across the whole data array
  int warp_limit = DIV_ROUNDUP(blockDim.x,CUDA_WARP_WIDTH);
  if(threadIdx.x < CUDA_WARP_WIDTH){
    warp_var = (float) (threadIdx.x < warp_limit ? dwarp_variances[threadIdx.x] : 0.0);
    // *NOTA BENE*: if CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH > CUDA_WARP_WIDTH this reduce will be incomplete!!
    // As of 2019 this is not the case for any CUDA device (typically 1028/32, but can be 512/32 on older GPU cards).
    __syncwarp();
    warp_var = warpReduceSum(warp_var);
    if(! threadIdx.x){
      threadblock_variances[blockIdx.x] = warp_var;
      //printf("Block final %d variance (within %d): %f\n", blockIdx.x, (int) data_length, warp_var);
      // If this is the last reduction, calculate the std dev from the variance so we don't have to copy the variance back to the CPU to do so
      if(gridDim.x == 1) 
        threadblock_variances[0] = (float) sqrt(threadblock_variances[0]/orig_data_length);
    }
  }
}

// Host function that gets the znormalized stats of a given set of data
// data - the data we are getting the stats from
// data_length - the length of the input data
// mean - the mean of the data to be calculated
// stddev - the standard deviation of the data to be calculated
// min - the minimum value of the data to be obtained
// max - the max value of the data to be obtained
// stream - CUDA kernel stream
template <class T>
__host__
void
get_znorm_stats(T *data, long data_length, float *mean, float *stddev, T *min, T *max, cudaStream_t stream){

  if(data_length < 2){
    std::cerr << "Provided data length was " << data_length << ", from which Z-normalization stats cannot be calculated, aborting." << std::endl;
    exit(23);
  }

  // TODO: merge to a single cudaMalloc()
  int num_threadblocks = DIV_ROUNDUP(data_length, CUDA_THREADBLOCK_MAX_THREADS);
  float *threadblock_means;
  T *threadblock_mins;
  T *threadblock_maxs;
  cudaMalloc(&threadblock_means, sizeof(float)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);               CUERR("Allocating device memory for query Z-norm threadblock means");
  cudaMalloc(&threadblock_mins, sizeof(T)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);                CUERR("Allocating device memory for query Z-norm threadblock mins");
  cudaMalloc(&threadblock_maxs, sizeof(T)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);                CUERR("Allocating device memory for query Z-norm threadblock maxs");
  float *threadblock_results;
  // There is a chance that these variances sums could get really big for very long datasets, TODO we may want to store into a double rather than float if QTYPE = float
  cudaMalloc(&threadblock_results, sizeof(float)*num_threadblocks);                CUERR("Allocating device memory for query Z-norm threadblock variances");

  dim3 grid(num_threadblocks, 1, 1);
  int req_threadblock_shared_memory = CUDA_THREADBLOCK_MAX_THREADS*sizeof(T)+(sizeof(T)*2+sizeof(float)+2)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
  mean_min_max<T><<<grid,CUDA_THREADBLOCK_MAX_THREADS,req_threadblock_shared_memory,stream>>>(data, data_length, threadblock_means, threadblock_mins, threadblock_maxs); CUERR("Calculating data mean/min/max");

  // Reduce across threadblock results
  while(num_threadblocks > 1){
    grid.x = DIV_ROUNDUP(num_threadblocks,CUDA_THREADBLOCK_MAX_THREADS);
    int threads = num_threadblocks > CUDA_THREADBLOCK_MAX_THREADS ? CUDA_THREADBLOCK_MAX_THREADS : num_threadblocks;
	req_threadblock_shared_memory = sizeof(float)*CUDA_WARP_WIDTH + 2*num_threadblocks*sizeof(T);
    mean_min_max_float<T><<<grid,threads,req_threadblock_shared_memory,stream>>>(threadblock_means, num_threadblocks, threadblock_means, threadblock_mins, threadblock_maxs); CUERR("Reducing calculated data mean/min/max");
    num_threadblocks = grid.x;
  }
  cudaMemcpyAsync(min, threadblock_mins, sizeof(T), cudaMemcpyDeviceToDevice, stream);             CUERR("Launching copy of calculated min within GPU global memory");
  cudaMemcpyAsync(max, threadblock_maxs, sizeof(T), cudaMemcpyDeviceToDevice, stream);             CUERR("Launching copy of calculated max within GPU global memory");

  num_threadblocks = DIV_ROUNDUP(data_length, CUDA_THREADBLOCK_MAX_THREADS);
  req_threadblock_shared_memory = sizeof(float)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
  grid.x = num_threadblocks;
  variance<T><<<grid,CUDA_THREADBLOCK_MAX_THREADS,req_threadblock_shared_memory,stream>>>(data, data_length, data_length, threadblock_means, threadblock_results); CUERR("Calculating data variance");
  // Reduce across threadblock results
  req_threadblock_shared_memory = sizeof(float)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
  while(num_threadblocks > 1){
    grid.x = DIV_ROUNDUP(num_threadblocks,CUDA_THREADBLOCK_MAX_THREADS);
    int threads = num_threadblocks > CUDA_THREADBLOCK_MAX_THREADS ? CUDA_THREADBLOCK_MAX_THREADS : num_threadblocks;
    variance_float<<<grid,threads,req_threadblock_shared_memory,stream>>>(threadblock_results, num_threadblocks, data_length, threadblock_means, 
                                                                     threadblock_results); CUERR("Reducing calculated data variance");
    num_threadblocks = grid.x;
  }
  cudaMemcpyAsync(mean, threadblock_means, sizeof(float), cudaMemcpyDeviceToDevice, stream);                 CUERR("Launching copy of calculated mean within GPU global memory");
  cudaMemcpyAsync(stddev, threadblock_results, sizeof(float), cudaMemcpyDeviceToDevice, stream);             CUERR("Launching copy of calculated std dev within GPU global memory");

  // TODO: Synchronizes on the first cudaFree()... move to a thread?
  cudaFree(threadblock_results);                                              CUERR("Freeing device memory for query Z-norm threadblock variances");
  cudaFree(threadblock_means);                                                CUERR("Freeing device memory for query Z-norm threadblock means");
  cudaFree(threadblock_mins);                                                 CUERR("Freeing device memory for query Z-norm threadblock mins");
  cudaFree(threadblock_maxs);                                                 CUERR("Freeing device memory for query Z-norm threadblock maxs");
}

// Hides the kernel launch details and inherent asynchronicity from the CUDA-agnotic caller.
// Note that it helps to speed up the transfer if the memory for *subject is page-locked on the CPU side.
// Host function that loads the subject onto the GPU
// subject - the subject to be loaded
// subject_std - subject standard deviation values
// subject_length - the length of the given subject
// use_std - flag for using subject standard deviation values
// stream - CUDA kernel stream
__host__
void 
load_subject(QTYPE *subject, QTYPE *subject_std, long subject_length, int use_std, cudaStream_t stream=0){
	if((subject && subject_length > 0) && ((use_std != 0) == (subject_std != 0))){
        // It turns out that cudaMalloc()s are pretty expensive (~1ms) so just allocate once (hence the +4) and stuff all the extra variables in that space at the end
        cudaMalloc(&Dsubject, sizeof(QTYPE)*(subject_length+10));                       CUERR("Allocating GPU memory for subject and its stats")
        cudaMemcpyToSymbolAsync(::Tsubject_length, &subject_length, sizeof(long), 0, cudaMemcpyHostToDevice, stream);         CUERR("Copying subject length from CPU to GPU memory")

        // The expensive step of copying a potentially large amount of data across the PCI bus...at least it's all in one chunk for efficiency.
        cudaMemcpyAsync(Dsubject, subject, sizeof(QTYPE)*subject_length, cudaMemcpyHostToDevice, stream); CUERR("Copying subject from CPU to GPU memory")
		
		if(use_std) {
			cudaMalloc(&Dsubject_std, sizeof(QTYPE)*(subject_length+10));
			cudaMemcpyAsync(Dsubject_std, subject_std, sizeof(QTYPE)*subject_length, cudaMemcpyHostToDevice, stream);
		}
		
		register int offset_mult = DIV_ROUNDUP(sizeof(float), sizeof(QTYPE));
        register float *mean_ptr = (float *) &Dsubject[subject_length];
        register float *stddev_ptr = (float *) &Dsubject[subject_length+offset_mult];
        register QTYPE *min_ptr = (QTYPE *) &Dsubject[subject_length+2*offset_mult];
        register QTYPE *max_ptr = (QTYPE *) &Dsubject[subject_length+2*offset_mult+1];

        get_znorm_stats(Dsubject, subject_length, mean_ptr, stddev_ptr, min_ptr, max_ptr, stream);

        cudaMemcpyToSymbolAsync(::Dsubject_mean, mean_ptr, sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);       CUERR("Copying subject mean from GPU global to constant memory")
        cudaMemcpyToSymbolAsync(::Dsubject_stddev, stddev_ptr, sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);   CUERR("Copying subject std dev from GPU global to constant memory")
        cudaMemcpyToSymbolAsync(::Dsubject_min, min_ptr, sizeof(QTYPE), 0, cudaMemcpyDeviceToDevice, stream);         CUERR("Copying subject min from GPU global to constant memory")
        cudaMemcpyToSymbolAsync(::Dsubject_max, max_ptr, sizeof(QTYPE), 0, cudaMemcpyDeviceToDevice, stream);         CUERR("Copying subject max from GPU global to constant memory")
	}
    else{
        subject_length = 0;
        cudaMemcpyToSymbolAsync(::Tsubject_length, &subject_length, sizeof(long), 0, cudaMemcpyHostToDevice, stream);         CUERR("Zeroing subject length in GPU constant memory")
    }
    cudaStreamSynchronize(stream);	CUERR("Synchronizing stream after loading the subject to GPU")
}

// Funtion that gets the Dsubject pointer for testing
// returns the Dsubject pointer
QTYPE* get_subject_pointer(){
	return Dsubject;
}

// Funtion that gets the Dsubject standard deviation pointer for testing
// returns the Dsubject_std pointer
QTYPE* get_subject_std_pointer(){
	return Dsubject_std;
}

// Z-normalize the query segmentation results (now in global constant memory) *all together*.
template <class T>
__global__
void
query_znorm(int offset, size_t length, float *data_mean, float *data_stddev, T *min, T *max, int normalization_mode, cudaStream_t stream){

      // TODO: There is a small chance that our quasi-z-normalization of a low dynamic range data stream (typically query) against a high dynamic
      // range target (subject) will cause the data stream's extreme values to shift or scale outside the bounds of QTYPE.  If this happens,
      // we will cap those transformed data stream values to QTYPE's min or max and output a warning.

      int pos = offset+blockIdx.x*blockDim.x+threadIdx.x;
      float mean = *data_mean;
	  // if(threadIdx.x == 0 && normalization_mode != NO_ZNORM){
        // printf("Data mean in znorm: %f and Sub mean: %f\n", mean, Dsubject_mean);
	    // printf("Data std in znorm: %f and Sub std: %f\n", *data_stddev, Dsubject_stddev);
		// printf("Min: %f and Max: %f\n", *min, *max);
      // }
      if(pos < offset+length){ // Coalesced global mem reads and writes
		// printf("pos: %i\n", pos);
		// printf("Gquery[pos]: %f\n", Gquery[pos]);
		if(normalization_mode != NO_ZNORM){
			Gquery[pos] = (QTYPE) ((Gquery[pos]-mean)/(*data_stddev)*Dsubject_stddev+Dsubject_mean);
			Gquery_std[pos] = (QTYPE) (Gquery_std[pos]/(*data_stddev)*Dsubject_stddev);
		}
      }
}

__global__
void
welford_query_znorm(int offset, int length, float* mean, float* ssq, long total_values_znormalized, cudaStream_t stream){

      // TODO: There is a small chance that our quasi-z-normalization of a low dynamic range data stream (typically query) against a high dynamic
      // range target (subject) will cause the data stream's extreme values to shift or scale outside the bounds of QTYPE.  If this happens,
      // we will cap those transformed data stream values to QTYPE's min or max and output a warning.

	  // TODO: With only one thread active, this will run very slowly. See if there is any way to speed it up

      int tid = threadIdx.x;
	  int pos = tid;
	  int num_it = DIV_ROUNDUP(length, CUDA_THREADBLOCK_MAX_THREADS);
	  
	  __shared__ QTYPE shared[CUDA_THREADBLOCK_MAX_THREADS];
	  
	  // float mean = 0;
	  float oldmean = 0;
	  // float ssq = 0;
	  float stddev;
	  float sample;
	  
	  for(int it = 0; it < num_it; it++) {
		if(pos < length)
			shared[tid] = Gquery[pos];
	  
		if(tid == 0){ // Coalesced global mem reads and writes
			long  p = total_values_znormalized;
			for(int i = 0; i < CUDA_THREADBLOCK_MAX_THREADS && p < total_values_znormalized + length; i++, p++) {
				sample = float(shared[i]);
				oldmean = (*mean);
				(*mean) += (sample-(*mean))/(p+1);
				(*ssq) += (sample-(*mean))*(sample-oldmean);
				stddev = (p > 0) ? sqrt((*ssq)/(p)) : 1; // TODO unroll this to avoid the if statement
				Gquery[p] = (QTYPE) ((sample-(*mean))/(stddev)*Dsubject_stddev+Dsubject_mean);
				Gquery_std[p] = (QTYPE) (Gquery_std[p]/(stddev)*Dsubject_stddev);
			}
		}
		pos += 1024;
      }
}

/* Hard coded full 10 point query and subject Euclidean distance White-Neely step DTW with Sakoe-Chiba band of 5.  
   It shares subject values across threadblock via L1 cache (shared memory) and query values via L1 cache too (constant memory). 
   Size of subject is limited to max(signed int) since CUDA supplied variables in blockIdx.x*blockDim.x+threadIdx.x 
   job index calculation is signed integer range bound. Query offset as multiple of ten is inferred by job index modulus subject length. 
   There are 24 4-byte registers per kernel thread.
*/

/* Note that this GPU-based DTW method assumes the Subject and Query are already z-normalized. Set up to run multiple DTWs in a thread block on a single CUDA multi-processor. */

// Let's make a function dist() for the Euclidean distance (L2 norm, not to be confused with L2 cache), just do the square w/o sqrt() as relative rank matters, not final value
#define std_dist(query_i,subject_j,qstd_i,std_i) ((query_i-subject_j)*(query_i-subject_j) + (qstd_i-std_i)*(qstd_i-std_i))
#define dist(query_i,subject_j) ((query_i-subject_j)*(query_i-subject_j))

/// MINIDTW_STRIDE must be power of 2
template<int soft_dtw_size, int soft_dtw_warp>
__global__
void soft_dtw_std(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances){
    const int soft_dtw_block_dim = CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE;
	const int soft_dtw_shared_it_float = soft_dtw_size;
	const int soft_dtw_stride = 1+soft_dtw_warp+soft_dtw_warp;

	// calculate jobid and subject/query sample locations for the thread
	// work with padding so that subject length is always CUDA_THREADBLOCK_MAX_THREADS ie. always fills the warp
	long long jobid = (long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x;
    long long eff_subject_length = CUDA_THREADBLOCK_MAX_THREADS*(DIV_ROUNDUP(((long long)Tsubject_length),CUDA_THREADBLOCK_MAX_THREADS));
	long long strided_jobid = jobid*MINIDTW_STRIDE;
	long long subject_offset = strided_jobid%eff_subject_length;
    int query_offset = int(strided_jobid/eff_subject_length*soft_dtw_size);

	// shared memory in 3 parts, one for storing subject, one for results of minidtws, one for indices for each thread when calculating min
	__shared__ char shared[sizeof(QTYPE)*(CUDA_THREADBLOCK_MAX_THREADS + soft_dtw_size)*2 + sizeof(QTYPE_ACC)*soft_dtw_block_dim + sizeof(long long)*soft_dtw_block_dim/2 + 32]; 
	QTYPE *subjects = (QTYPE *) &shared[0];
	size_t temp_unpadded = sizeof(QTYPE)*(CUDA_THREADBLOCK_MAX_THREADS + soft_dtw_size);
	size_t temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE *subject_stds = (QTYPE *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE)*(CUDA_THREADBLOCK_MAX_THREADS + soft_dtw_size);
	temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE_ACC *dtw_results = (QTYPE_ACC *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE_ACC)*soft_dtw_block_dim;
	temp_padded = temp_unpadded + 16 - temp_unpadded%16;
	long long *idata = (long long *) &shared[temp_padded];

    if(subject_offset < (long long) Tsubject_length - soft_dtw_size){ // if valid position (block is padded to CUDA_THREADBLOCK_MAX_THREADS)
   
    // Load the bits of the subject (from texture memory) that this thread will use into shared memory (L1 cache)
    //if(sizeof(QTYPE) == 4){
	#pragma unroll
	for(int i = 0; i < soft_dtw_shared_it_float; i++) {
		subjects[threadIdx.x + i] = subject[subject_offset + i];
	}
	#pragma unroll
	for(int i = 0; i < soft_dtw_shared_it_float; i++) {
		subject_stds[threadIdx.x + i] = subject_std[subject_offset + i];
	}
	
	// registers that will carry intermediate results
	// ideally these are in registers, but not always, may be sent out to local mem
	QTYPE q;
	QTYPE qstd;
	QTYPE s[soft_dtw_size];
	QTYPE std[soft_dtw_size];
    QTYPE_ACC p[soft_dtw_stride];
 
    // load each subject value into register just-in-time to avoid memory bottleneck on warp start
    #pragma unroll
	for(int i = 0; i < soft_dtw_stride; i++) {
		s[i] = subjects[threadIdx.x + i];
	}
	
	#pragma unroll
	for(int i = 0; i < soft_dtw_stride; i++) {
		std[i] = subject_stds[threadIdx.x + i];
	}

	#pragma unroll
	for(int query_pos = 0; query_pos < soft_dtw_size; query_pos++) {
		int load_s = query_pos + soft_dtw_stride;
		q = Gquery[query_offset + query_pos];
		qstd = Gquery_std[query_offset + query_pos];
		QTYPE_ACC previous_p = 0;
		if(load_s < soft_dtw_size) {
			s[load_s] = subjects[threadIdx.x + load_s];
			std[load_s] = subject_stds[threadIdx.x + load_s];
		}
		int stride_pos = query_pos - soft_dtw_warp;
		#pragma unroll
		for(int i = 0; i < soft_dtw_size; i++) {
			if(i < soft_dtw_stride) {
			if(stride_pos >= 0 && stride_pos < soft_dtw_size) {
				bool at_left = stride_pos == 0;
				bool at_bottom = query_pos == 0;
				if(stride_pos != 0 && query_pos != 0) {
					if(i == 0) {
						previous_p = dtwmin(p[i], p[i + 1]);
					}
					else if(i == soft_dtw_stride-1) {
						previous_p = dtwmin(p[i], previous_p);
					}
					else {
						previous_p = dtwmin(p[i + 1], previous_p);	
						previous_p = dtwmin(p[i], previous_p);
					}
				}
				else if(at_left && !at_bottom) {
					previous_p = p[i + 1];
				}
				p[i] = std_dist(q,s[stride_pos],qstd,std[stride_pos]) + previous_p;
				previous_p = p[i];
			}
			stride_pos++;
			}
		}
	}
	
	dtw_results[threadIdx.x] = p[soft_dtw_warp];

    } else {
	
	dtw_results[threadIdx.x] = DTW_MAX;
	
	}
	
	__syncthreads();
	
	QTYPE_ACC min = DTW_MAX;
	
	long long index = -1;
	
	// call find min function, only use of shared idata (for inter-thread communication), returns min and index
	block_findMin(dtw_results, idata, threadIdx.x, blockDim.x, strided_jobid, min, index);
	
	// store result, indexed by query first
	if(threadIdx.x == 0) {
		query_adjacent_candidates[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/soft_dtw_size] = index%eff_subject_length;
		query_adjacent_distances[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/soft_dtw_size] = min;
	}
}

template<int soft_dtw_size, int soft_dtw_warp>
__global__
void soft_dtw(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances){
    const int soft_dtw_block_dim = CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE;
	const int soft_dtw_shared_it_float = soft_dtw_size;
	const int soft_dtw_stride = 1+soft_dtw_warp+soft_dtw_warp;

	// calculate jobid and subject/query sample locations for the thread
	// work with padding so that subject length is always CUDA_THREADBLOCK_MAX_THREADS ie. always fills the warp
	long long jobid = (long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x;
    long long eff_subject_length = CUDA_THREADBLOCK_MAX_THREADS*(DIV_ROUNDUP(((long long)Tsubject_length),CUDA_THREADBLOCK_MAX_THREADS));
	long long strided_jobid = jobid*MINIDTW_STRIDE;
	long long subject_offset = strided_jobid%eff_subject_length;
    int query_offset = int(strided_jobid/eff_subject_length*soft_dtw_size);

	// shared memory in 3 parts, one for storing subject, one for results of minidtws, one for indices for each thread when calculating min
	__shared__ char shared[sizeof(QTYPE)*(CUDA_THREADBLOCK_MAX_THREADS + soft_dtw_size)*2 + sizeof(QTYPE_ACC)*soft_dtw_block_dim + sizeof(long long)*soft_dtw_block_dim/2 + 32]; 
	QTYPE *subjects = (QTYPE *) &shared[0];
	size_t temp_unpadded = sizeof(QTYPE)*(CUDA_THREADBLOCK_MAX_THREADS + soft_dtw_size);
	size_t temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE_ACC *dtw_results = (QTYPE_ACC *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE_ACC)*soft_dtw_block_dim;
	temp_padded = temp_unpadded + 16 - temp_unpadded%16;
	long long *idata = (long long *) &shared[temp_padded];

    if(subject_offset <= (long long) Tsubject_length - soft_dtw_size){ // if valid position (block is padded to CUDA_THREADBLOCK_MAX_THREADS)
   
    // Load the bits of the subject (from texture memory) that this thread will use into shared memory (L1 cache)
    //if(sizeof(QTYPE) == 4){
	#pragma unroll
	for(int i = 0; i < soft_dtw_shared_it_float; i++) {
		subjects[threadIdx.x + i] = subject[subject_offset + i];
	}
	
	// registers that will carry intermediate results
	// ideally these are in registers, but not always, may be sent out to local mem
	QTYPE q;
	QTYPE s[soft_dtw_size];
    QTYPE_ACC p[soft_dtw_stride];
 
    // load each subject value into register just-in-time to avoid memory bottleneck on warp start
    #pragma unroll
	for(int i = 0; i < soft_dtw_stride; i++) {
		s[i] = subjects[threadIdx.x + i];
	}

	#pragma unroll
	for(int query_pos = 0; query_pos < soft_dtw_size; query_pos++) {
		int load_s = query_pos + soft_dtw_stride;
		q = Gquery[query_offset + query_pos];
		QTYPE_ACC previous_p = 0;
		if(load_s < soft_dtw_size) {
			s[load_s] = subjects[threadIdx.x + load_s];
		}
		int stride_pos = query_pos - soft_dtw_warp;
		#pragma unroll
		for(int i = 0; i < soft_dtw_size; i++) {
			if(i < soft_dtw_stride) {
			if(stride_pos >= 0 && stride_pos < soft_dtw_size) {
				bool at_left = stride_pos == 0;
				bool at_bottom = query_pos == 0;
				if(stride_pos != 0 && query_pos != 0) {
					if(i == 0) {
						previous_p = dtwmin(p[i], p[i + 1]);
					}
					else if(i == soft_dtw_stride-1) {
						previous_p = dtwmin(p[i], previous_p);
					}
					else {
						previous_p = dtwmin(p[i + 1], previous_p);	
						previous_p = dtwmin(p[i], previous_p);
					}
				}
				else if(at_left && !at_bottom) {
					previous_p = p[i + 1];
				}
				p[i] = dist(q,s[stride_pos]) + previous_p;
				previous_p = p[i];
			}
			stride_pos++;
			}
		}
	}
	
	dtw_results[threadIdx.x] = p[soft_dtw_warp];

    } else {
	
	dtw_results[threadIdx.x] = DTW_MAX;
	
	}
	
	__syncthreads();
	
	QTYPE_ACC min = DTW_MAX;
	
	long long index = -1;
	
	// call find min function, only use of shared idata (for inter-thread communication), returns min and index
	block_findMin(dtw_results, idata, threadIdx.x, blockDim.x, strided_jobid, min, index);
	
	// store result, indexed by query first
	if(threadIdx.x == 0) {
		query_adjacent_candidates[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/soft_dtw_size] = index%eff_subject_length;
		query_adjacent_distances[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/soft_dtw_size] = min;
	}
}

__global__
void hard_dtw_std(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances){
    #define soft_dtw_block_dim (CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE)
	
	// calculate jobid and subject/query sample locations for the thread
	// work with padding so that subject length is always 1024 ie. always fills the warp
	long long jobid = (long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x;
	long long eff_subject_length = CUDA_THREADBLOCK_MAX_THREADS*(DIV_ROUNDUP(((long long)Tsubject_length),CUDA_THREADBLOCK_MAX_THREADS));
	long long strided_jobid = jobid*MINIDTW_STRIDE;
    long long subject_offset = strided_jobid%eff_subject_length;
    int query_offset = int(strided_jobid/eff_subject_length*10);

    // 11 registers for 10-point DTW (need two times the warp band width of 5 for all valid step combinations, and a query value)
    // Plus we've decided to put the subject values in registers for speed
    QTYPE q, qstd, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, std0, std1, std2, std3, std4, std5, std6, std7, std8, std9;
    QTYPE_ACC p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;

	// shared memory in 3 parts, one for storing subject, one for results of minidtws, one for indices for each thread when calculating min
    __shared__ char shared[2*sizeof(QTYPE)*(1034) + sizeof(QTYPE_ACC)*soft_dtw_block_dim + sizeof(long long)*soft_dtw_block_dim/2 + 32];
	QTYPE * s = (QTYPE *) &shared[0];
	size_t temp_unpadded = sizeof(QTYPE)*(1034);
	size_t temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE * std = (QTYPE *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE_ACC)*(1034);
	temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE_ACC *dtw_results = (QTYPE_ACC *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE_ACC)*soft_dtw_block_dim;
	temp_padded = temp_unpadded + 16 - temp_unpadded%16;
	long long *idata = (long long *) &shared[temp_padded];

    if(subject_offset < Tsubject_length - 10){ // if valid position (always padded to 1024)
   
    // Load the bits of the subject (from texture memory) that this thread will use into shared memory (L1 cache)
    //if(sizeof(QTYPE) == 4){
    s[threadIdx.x] = subject[subject_offset];
    s[threadIdx.x+1] = subject[subject_offset+1];
    s[threadIdx.x+2] = subject[subject_offset+2];
    s[threadIdx.x+3] = subject[subject_offset+3];
    s[threadIdx.x+4] = subject[subject_offset+4];
    s[threadIdx.x+5] = subject[subject_offset+5];
    s[threadIdx.x+6] = subject[subject_offset+6];
    s[threadIdx.x+7] = subject[subject_offset+7];
    s[threadIdx.x+8] = subject[subject_offset+8];
    s[threadIdx.x+9] = subject[subject_offset+9];
	std[threadIdx.x] = subject_std[subject_offset];
    std[threadIdx.x+1] = subject_std[subject_offset+1];
    std[threadIdx.x+2] = subject_std[subject_offset+2];
    std[threadIdx.x+3] = subject_std[subject_offset+3];
    std[threadIdx.x+4] = subject_std[subject_offset+4];
    std[threadIdx.x+5] = subject_std[subject_offset+5];
    std[threadIdx.x+6] = subject_std[subject_offset+6];
    std[threadIdx.x+7] = subject_std[subject_offset+7];
    std[threadIdx.x+8] = subject_std[subject_offset+8];
    std[threadIdx.x+9] = subject_std[subject_offset+9];
    /*}
    else if(sizeof(QTYPE) == 2){ // only need half as many fetches since texture fetch is always 4 bytes
      s[threadIdx.x] = subject[subject_offset];
      s[threadIdx.x+2] = subject[subject_offset+1];
      s[threadIdx.x+4] = subject[subject_offset+2];
      s[threadIdx.x+6] = subject[subject_offset+3];
      s[threadIdx.x+8] = subject[subject_offset+4];
    }
    else{
      s[threadIdx.x] = subject[subject_offset];
      s[threadIdx.x+4] = subject[subject_offset+1];
      s[threadIdx.x+8] = subject[subject_offset+2];
      // Uh-oh...if we made the DTW 12 point we could support 1 byte query values...but then we'd need >1 byte distance accumulators..TODO
      //std::cerr << "QTYPE was neither 4 nor 2 bytes in size (was " << sizeof(QTYPE) << "), aborting.", std::endl;
      return;
    }*/
 
    // load each subject value into register just-in-time to avoid memory bottleneck on warp start
    s0 = s[threadIdx.x];
    s1 = s[threadIdx.x+1];
    s2 = s[threadIdx.x+2];
    s3 = s[threadIdx.x+3];
    s4 = s[threadIdx.x+4];
	std0 = std[threadIdx.x];
    std1 = std[threadIdx.x+1];
    std2 = std[threadIdx.x+2];
    std3 = std[threadIdx.x+3];
    std4 = std[threadIdx.x+4];

    // Step direction labels here imagine DTW path cost computation where the query is on the X axis, and the subject is on the Y axis in a Cartesian configuration
    // 1st query position costs
    q = Gquery[query_offset];
	qstd = Gquery_std[query_offset];
    p0 = std_dist(q,s0,qstd,std0);
    p1 = std_dist(q,s1,qstd,std1) + p0; // p0 -> p1 is up = a subject deletion
    p2 = std_dist(q,s2,qstd,std2) + p1;
    p3 = std_dist(q,s3,qstd,std3) + p2;
    p4 = std_dist(q,s4,qstd,std4) + p3;
	
    // 2nd query column
    q = Gquery[query_offset+1];
	qstd = Gquery_std[query_offset+1];
    p5 = std_dist(q,s0,qstd,std0) + p0;
    p6 = std_dist(q,s1,qstd,std1) + dtwmin(p5, dtwmin(p0, p1)); // This match + cumulative cost so far from step min(up, min(diagonal,right)) to get here ...
                                            // where right (p1 -> p6) is a query deletion and diagonal (p0 -> p6) is a match
    p7 = std_dist(q,s2,qstd,std2) + dtwmin(p6, dtwmin(p1, p2)); 
    p8 = std_dist(q,s3,qstd,std3) + dtwmin(p7, dtwmin(p2, p3));
    p9 = std_dist(q,s4,qstd,std4) + dtwmin(p8, dtwmin(p3, p4));

    // 3rd query column (recycle registers from two columns back from now on)
    // shift the set of valid paths up one subject index as a quasi-Sakoe-Chiba banding for the DTW path
    s5 = s[threadIdx.x+5];
	std5 = std[threadIdx.x+5];
    q = Gquery[query_offset+2];
	qstd = Gquery_std[query_offset+2];
    p0 = std_dist(q,s1,qstd,std1) + dtwmin(p5, p6); // this match + cumulative cost so far from step min(diagonal,up) as we are at the lower edge of the shifted band
    p1 = std_dist(q,s2,qstd,std2) + dtwmin(p0, dtwmin(p6, p7));
    p2 = std_dist(q,s3,qstd,std3) + dtwmin(p1, dtwmin(p7, p8));
    p3 = std_dist(q,s4,qstd,std4) + dtwmin(p2, dtwmin(p8, p9));
    p4 = std_dist(q,s5,qstd,std5) + dtwmin(p3, p9);

    // 4th query column
    q = Gquery[query_offset+3];
	qstd = Gquery_std[query_offset+3];
    p5 = std_dist(q,s1,qstd,std1) + p0;
    p6 = std_dist(q,s2,qstd,std2) + dtwmin(p5, dtwmin(p0, p1));
    p7 = std_dist(q,s3,qstd,std3) + dtwmin(p6, dtwmin(p1, p2));
    p8 = std_dist(q,s4,qstd,std4) + dtwmin(p7, dtwmin(p2, p3));
    p9 = std_dist(q,s5,qstd,std5) + dtwmin(p8, dtwmin(p3, p4));

    // 5th query column (band jogs one more spot up the matrix)
    s6 = s[threadIdx.x+6];
	std6 = std[threadIdx.x+6];
    q = Gquery[query_offset+4];
	qstd = Gquery_std[query_offset+4];
    p0 = std_dist(q,s2,qstd,std2) + dtwmin(p5, p6);
    p1 = std_dist(q,s3,qstd,std3) + dtwmin(p0, dtwmin(p6, p7));
    p2 = std_dist(q,s4,qstd,std4) + dtwmin(p1, dtwmin(p7, p8));
    p3 = std_dist(q,s5,qstd,std5) + dtwmin(p2, dtwmin(p8, p9));
    p4 = std_dist(q,s6,qstd,std6) + dtwmin(p3, p9);
	
    // 6th query column (band jog)
    s7 = s[threadIdx.x+7];
	std7 = s[threadIdx.x+7];
    q = Gquery[query_offset+5];
	qstd = Gquery_std[query_offset+5];
    p5 = std_dist(q,s3,qstd,std3) + dtwmin(p0, p1);
    p6 = std_dist(q,s4,qstd,std4) + dtwmin(p5, dtwmin(p1, p2));
    p7 = std_dist(q,s5,qstd,std5) + dtwmin(p6, dtwmin(p2, p3));
    p8 = std_dist(q,s6,qstd,std6) + dtwmin(p7, dtwmin(p3, p4));
    p9 = std_dist(q,s7,qstd,std7) + dtwmin(p8, p4);

    // 7th query column
    q = Gquery[query_offset+6];
	qstd = Gquery_std[query_offset+6];
    p0 = std_dist(q,s3,qstd,std3) + p5;
    p1 = std_dist(q,s4,qstd,std4) + dtwmin(p0, dtwmin(p5, p6));
    p2 = std_dist(q,s5,qstd,std5) + dtwmin(p1, dtwmin(p6, p7));
    p3 = std_dist(q,s6,qstd,std6) + dtwmin(p2, dtwmin(p7, p8));
	p4 = std_dist(q,s7,qstd,std7) + dtwmin(p3, dtwmin(p8, p9));

    // 8th query column (band jog)
    s8 = s[threadIdx.x+8];
	std8 = std[threadIdx.x+8];
    q = Gquery[query_offset+7];
	qstd = Gquery_std[query_offset+7];
    p5 = std_dist(q,s4,qstd,std4) + dtwmin(p0, p1);
    p6 = std_dist(q,s5,qstd,std5) + dtwmin(p5, dtwmin(p1, p2));
    p7 = std_dist(q,s6,qstd,std6) + dtwmin(p6, dtwmin(p2, p3));
    p8 = std_dist(q,s7,qstd,std7) + dtwmin(p7, dtwmin(p3, p4));
    p9 = std_dist(q,s8,qstd,std8) + dtwmin(p8, p4);

    // 9th query column (final band jog)
    s9 = s[threadIdx.x+9];
	std9 = std[threadIdx.x+9];
    q = Gquery[query_offset+8];
	qstd = Gquery_std[query_offset+8];
    p0 = std_dist(q,s5,qstd,std5) + dtwmin(p5, p6);
    p1 = std_dist(q,s6,qstd,std6) + dtwmin(p0, dtwmin(p6, p7));
    p2 = std_dist(q,s7,qstd,std7) + dtwmin(p1, dtwmin(p7, p8));
    p3 = std_dist(q,s8,qstd,std8) + dtwmin(p2, dtwmin(p8, p9)); 
    p4 = std_dist(q,s9,qstd,std9) + dtwmin(p3, p9);
	
    // 10th and final query column 
    q = Gquery[query_offset+9];
	qstd = Gquery_std[query_offset+9];
    p5 = std_dist(q,s5,qstd,std5) + p0;
    p6 = std_dist(q,s6,qstd,std6) + dtwmin(p5, dtwmin(p0, p1));
    p7 = std_dist(q,s7,qstd,std7) + dtwmin(p6, dtwmin(p1, p2));
    p8 = std_dist(q,s8,qstd,std8) + dtwmin(p7, dtwmin(p2, p3));
    // Warp path must end in the upper right corner (don't bother assigning to p9).
	
    dtw_results[threadIdx.x] = std_dist(q,s9,qstd,std9) + dtwmin(p8, dtwmin(p3, p4));

    } else {
	
	dtw_results[threadIdx.x] = DTW_MAX;
	
	}
	
	__syncthreads();
	
	QTYPE_ACC min = DTW_MAX;
	
	long long index = -1;
	
	// call find min function, only use of shared idata (for inter-thread communication), returns min and index
	block_findMin(dtw_results, idata, threadIdx.x, blockDim.x, strided_jobid, min, index);
	
	// store result, indexed by query first
	if(threadIdx.x == 0) {
		query_adjacent_candidates[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/10] = index%eff_subject_length;
		query_adjacent_distances[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/10] = min;
	}
}

__global__
void hard_dtw(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, int minidtw_size){
    #define soft_dtw_block_dim (CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE)
	
	// calculate jobid and subject/query sample locations for the thread
	// work with padding so that subject length is always 1024 ie. always fills the warp
	long long jobid = (long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x;
	long long eff_subject_length = CUDA_THREADBLOCK_MAX_THREADS*(DIV_ROUNDUP(((long long)Tsubject_length),CUDA_THREADBLOCK_MAX_THREADS));
	long long strided_jobid = jobid*MINIDTW_STRIDE;
    long long subject_offset = strided_jobid%eff_subject_length;
    int query_offset = int(strided_jobid/eff_subject_length*minidtw_size);

    // 11 registers for 10-point DTW (need two times the warp band width of 5 for all valid step combinations, and a query value)
    // Plus we've decided to put the subject values in registers for speed
    QTYPE q, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
    QTYPE_ACC p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;

	// shared memory in 3 parts, one for storing subject, one for results of minidtws, one for indices for each thread when calculating min
    __shared__ char shared[sizeof(QTYPE)*(1034) + sizeof(QTYPE_ACC)*soft_dtw_block_dim + sizeof(long long)*soft_dtw_block_dim/2 + 24];
	QTYPE * s = (QTYPE *) &shared[0];
	size_t temp_unpadded = sizeof(QTYPE)*(1034);
	size_t temp_padded = temp_unpadded + 8 - temp_unpadded%8;
	QTYPE_ACC *dtw_results = (QTYPE_ACC *) &shared[temp_padded];
	temp_unpadded = temp_padded + sizeof(QTYPE_ACC)*soft_dtw_block_dim;
	temp_padded = temp_unpadded + 16 - temp_unpadded%16;
	long long *idata = (long long *) &shared[temp_padded];

    if(subject_offset <= Tsubject_length - 10 && query_offset <= Gquery_length - 10){ // if valid position (always padded to CUDA_THREADBLOCK_MAX_THREADS)
   
		// Load the bits of the subject (from texture memory) that this thread will use into shared memory (L1 cache)
		//if(sizeof(QTYPE) == 4){
		s[threadIdx.x] = subject[subject_offset];
		s[threadIdx.x+1] = subject[subject_offset+1];
		s[threadIdx.x+2] = subject[subject_offset+2];
		s[threadIdx.x+3] = subject[subject_offset+3];
		s[threadIdx.x+4] = subject[subject_offset+4];
		s[threadIdx.x+5] = subject[subject_offset+5];
		s[threadIdx.x+6] = subject[subject_offset+6];
		s[threadIdx.x+7] = subject[subject_offset+7];
		s[threadIdx.x+8] = subject[subject_offset+8];
		s[threadIdx.x+9] = subject[subject_offset+9];
		/*}
		else if(sizeof(QTYPE) == 2){ // only need half as many fetches since texture fetch is always 4 bytes
		s[threadIdx.x] = subject[subject_offset];
		s[threadIdx.x+2] = subject[subject_offset+1];
		s[threadIdx.x+4] = subject[subject_offset+2];
		s[threadIdx.x+6] = subject[subject_offset+3];
		s[threadIdx.x+8] = subject[subject_offset+4];
		}
		else{
		s[threadIdx.x] = subject[subject_offset];
		s[threadIdx.x+4] = subject[subject_offset+1];
		s[threadIdx.x+8] = subject[subject_offset+2];
		// Uh-oh...if we made the DTW 12 point we could support 1 byte query values...but then we'd need >1 byte distance accumulators..TODO
		//std::cerr << "QTYPE was neither 4 nor 2 bytes in size (was " << sizeof(QTYPE) << "), aborting.", std::endl;
		return;
		}*/
	
		// load each subject value into register just-in-time to avoid memory bottleneck on warp start
		s0 = s[threadIdx.x];
		s1 = s[threadIdx.x+1];
		s2 = s[threadIdx.x+2];
		s3 = s[threadIdx.x+3];
		s4 = s[threadIdx.x+4];
	
	
		// Step direction labels here imagine DTW path cost computation where the query is on the X axis, and the subject is on the Y axis in a Cartesian configuration
		// 1st query position costs
		q = Gquery[query_offset];
		p0 = dist(q,s0);
		p1 = dist(q,s1) + p0; // p0 -> p1 is up = a subject deletion
		p2 = dist(q,s2) + p1;
		p3 = dist(q,s3) + p2;
		p4 = dist(q,s4) + p3;
		
		// 2nd query column
		q = Gquery[query_offset+1];
		p5 = dist(q,s0) + p0;
		p6 = dist(q,s1) + dtwmin(p5, dtwmin(p0, p1)); // This match + cumulative cost so far from step min(up, min(diagonal,right)) to get here ...
												// where right (p1 -> p6) is a query deletion and diagonal (p0 -> p6) is a match
		p7 = dist(q,s2) + dtwmin(p6, dtwmin(p1, p2)); 
		p8 = dist(q,s3) + dtwmin(p7, dtwmin(p2, p3));
		p9 = dist(q,s4) + dtwmin(p8, dtwmin(p3, p4));
	
		// 3rd query column (recycle registers from two columns back from now on)
		// shift the set of valid paths up one subject index as a quasi-Sakoe-Chiba banding for the DTW path
		s5 = s[threadIdx.x+5];
	
		q = Gquery[query_offset+2];
		p0 = dist(q,s1) + dtwmin(p5, p6); // this match + cumulative cost so far from step min(diagonal,up) as we are at the lower edge of the shifted band
		p1 = dist(q,s2) + dtwmin(p0, dtwmin(p6, p7));
		p2 = dist(q,s3) + dtwmin(p1, dtwmin(p7, p8));
		p3 = dist(q,s4) + dtwmin(p2, dtwmin(p8, p9));
		p4 = dist(q,s5) + dtwmin(p3, p9);
	
		// 4th query column
		q = Gquery[query_offset+3];
		p5 = dist(q,s1) + p0;
		p6 = dist(q,s2) + dtwmin(p5, dtwmin(p0, p1));
		p7 = dist(q,s3) + dtwmin(p6, dtwmin(p1, p2));
		p8 = dist(q,s4) + dtwmin(p7, dtwmin(p2, p3));
		p9 = dist(q,s5) + dtwmin(p8, dtwmin(p3, p4));
	
		// 5th query column (band jogs one more spot up the matrix)
		s6 = s[threadIdx.x+6];
		q = Gquery[query_offset+4];
		p0 = dist(q,s2) + dtwmin(p5, p6);
		p1 = dist(q,s3) + dtwmin(p0, dtwmin(p6, p7));
		p2 = dist(q,s4) + dtwmin(p1, dtwmin(p7, p8));
		p3 = dist(q,s5) + dtwmin(p2, dtwmin(p8, p9));
		p4 = dist(q,s6) + dtwmin(p3, p9);
	
		// 6th query column (band jog)
		s7 = s[threadIdx.x+7];
		q = Gquery[query_offset+5];
		p5 = dist(q,s3) + dtwmin(p0, p1);
		p6 = dist(q,s4) + dtwmin(p5, dtwmin(p1, p2));
		p7 = dist(q,s5) + dtwmin(p6, dtwmin(p2, p3));
		p8 = dist(q,s6) + dtwmin(p7, dtwmin(p3, p4));
		p9 = dist(q,s7) + dtwmin(p8, p4);
	
		// 7th query column
		q = Gquery[query_offset+6];
		p0 = dist(q,s3) + p5;
		p1 = dist(q,s4) + dtwmin(p0, dtwmin(p5, p6));
		p2 = dist(q,s5) + dtwmin(p1, dtwmin(p6, p7));
		p3 = dist(q,s6) + dtwmin(p2, dtwmin(p7, p8));
		p4 = dist(q,s7) + dtwmin(p3, dtwmin(p8, p9));
	
		// 8th query column (band jog)
		s8 = s[threadIdx.x+8];
		q = Gquery[query_offset+7];
		p5 = dist(q,s4) + dtwmin(p0, p1);
		p6 = dist(q,s5) + dtwmin(p5, dtwmin(p1, p2));
		p7 = dist(q,s6) + dtwmin(p6, dtwmin(p2, p3));
		p8 = dist(q,s7) + dtwmin(p7, dtwmin(p3, p4));
		p9 = dist(q,s8) + dtwmin(p8, p4);
	
		// 9th query column (final band jog)
		s9 = s[threadIdx.x+9];
		q = Gquery[query_offset+8];
		p0 = dist(q,s5) + dtwmin(p5, p6);
		p1 = dist(q,s6) + dtwmin(p0, dtwmin(p6, p7));
		p2 = dist(q,s7) + dtwmin(p1, dtwmin(p7, p8));
		p3 = dist(q,s8) + dtwmin(p2, dtwmin(p8, p9)); 
		p4 = dist(q,s9) + dtwmin(p3, p9);
	
		// 10th and final query column 
		q = Gquery[query_offset+9];
		p5 = dist(q,s5) + p0;
		p6 = dist(q,s6) + dtwmin(p5, dtwmin(p0, p1));
		p7 = dist(q,s7) + dtwmin(p6, dtwmin(p1, p2));
		p8 = dist(q,s8) + dtwmin(p7, dtwmin(p2, p3));
		// Warp path must end in the upper right corner (don't bother assigning to p9).
		
		dtw_results[threadIdx.x] = dist(q,s9) + dtwmin(p8, dtwmin(p3, p4));

    } else {
	
		dtw_results[threadIdx.x] = DTW_MAX;
	
	}
	
	__syncthreads();
	
	QTYPE_ACC min = DTW_MAX;
	
	long long index = -1;
	
	// call find min function, only use of shared idata (for inter-thread communication), returns min and index
	block_findMin(dtw_results, idata, threadIdx.x, blockDim.x, strided_jobid, min, index);
	
	// store result, indexed by query first
	if(threadIdx.x == 0) {
		query_adjacent_candidates[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/minidtw_size] = index%eff_subject_length;
		query_adjacent_distances[(subject_offset/CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices + query_offset/minidtw_size] = min;
	}
}

__inline__ __host__
void
soft_dtw_wrap(const dim3 griddim_dtw, const int threadblock_size_dtw, cudaStream_t stream, const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, const int minidtw_size, const int minidtw_warp, const int use_std){
  if(minidtw_size == 10 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<10,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<10,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 11 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<11,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<11,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 12 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<12,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<12,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 13 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<13,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<13,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 14 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<14,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<14,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 15 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<15,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<15,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 16 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<16,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<16,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 17 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<17,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<17,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 18 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<18,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<18,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 19 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<19,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<19,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 20 && minidtw_warp == 2) {
	if(use_std) {
	  soft_dtw_std<20,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<20,2><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 10 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<10,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<10,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 11 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<11,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<11,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 12 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<12,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<12,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 13 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<13,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<13,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 14 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<14,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<14,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 15 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<15,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<15,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 16 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<16,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<16,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 17 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<17,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<17,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 18 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<18,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<18,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 19 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<19,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<19,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 20 && minidtw_warp == 3) {
	if(use_std) {
	  soft_dtw_std<20,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<20,3><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 10 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<10,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<10,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 11 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<11,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<11,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 12 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<12,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<12,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 13 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<13,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<13,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 14 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<14,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<14,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 15 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<15,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<15,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 16 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<16,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<16,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 17 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<17,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<17,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 18 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<18,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<18,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 19 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<19,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<19,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  } else if(minidtw_size == 20 && minidtw_warp == 4) {
	if(use_std) {
	  soft_dtw_std<20,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
	} else {
	  soft_dtw<20,4><<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances); CUERR("Running DTW anchor distance calculations")
    }
  }
}

inline __device__
void thread_findMin(QTYPE *dtw_results, long long *idata, const int tid, const int block_dim, const long long strided_jobid, int thread_dim){
	if(tid < thread_dim && block_dim/2 >= thread_dim) {
		if (dtw_results[tid] > dtw_results[tid + thread_dim]){
			dtw_results[tid] = dtw_results[tid + thread_dim];
			if(block_dim/2 < 2*thread_dim){
				idata[tid] = strided_jobid + thread_dim;
			}
			else{
				idata[tid] = idata[tid + thread_dim];
			}
		} else if(block_dim/2 < 2*thread_dim){
			idata[tid] = strided_jobid;
		}
	}
	__syncthreads();
}

inline __device__
void warp_findMin(QTYPE *dtw_results, long long *idata, const int tid, const int block_dim, const long long strided_jobid, int warp_dim){
	if(block_dim/2 >= warp_dim){
		if (dtw_results[tid] > dtw_results[tid + warp_dim]){
			dtw_results[tid] = dtw_results[tid + warp_dim];
			if(block_dim/2 < 2*warp_dim){
				idata[tid] = strided_jobid + warp_dim;
			}
			else{
				idata[tid] = idata[tid + warp_dim];
			}
		}
		else if(block_dim/2 < 2*warp_dim){
			idata[tid] = strided_jobid;
		}
		__syncwarp();
	}
}

inline __device__
void block_findMin(QTYPE *dtw_results, long long *idata, const int tid, const int block_dim, const long long strided_jobid, QTYPE_ACC &min, long long &index) {
	
	#if MINIDTW_STRIDE == 1
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 512);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 256);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 128);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 64);

	#elif MINIDTW_STRIDE == 2
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 256);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 128);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 64);

	#elif MINIDTW_STRIDE == 4
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 128);
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 64);

	#elif MINIDTW_STRIDE == 8
		thread_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 64);

	#else
	#error "MINIDTW_STRIDE must be one of  {8, 4, 2, 1}"
	#endif
	if(tid < 32) {
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 32);
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 16);
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 8);
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 4);
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 2);
		warp_findMin(dtw_results, idata, tid, block_dim, strided_jobid, 1);
	}
	
	if(tid == 0) {
		min = dtw_results[0];
		index = idata[0];
	}
}

// Device side funtion that gets the membership of the indices observed in set_pairwise_bits based on their colinearity
// membership - the buffer that will store the membership results
// num_candidate_query_indices - the number of values to be storred in membership
// subject_indices - indices of the subject that were found to be candidates
// minidtw_size - length of comparison between query and subject within a mini-DTW iteration
// max_warp_proportion- proportion of length deviation allowed between query and subject in alignment
// global_mem - flag that indicates if the two above buffers are global memory or shared memory
template<class T>
__inline__ __device__ void set_membership_results(T* membership, const int num_candidate_query_indices, long long* subject_indices, const int minidtw_size, const float max_warp_proportion, bool global_mem){

	if(threadIdx.x == 0) {
		int member = 0;
		for(int i = 0; i < num_candidate_query_indices; i++) {
			if(membership[i] != -1) {
				continue;
			}
			long long base = subject_indices[i];
			int count = 0;
			for(int j = i+1; j < num_candidate_query_indices; j++) {
				if(membership[j] != -1)
					continue;
				int base_left = base+int(minidtw_size*float(j-i)*(1-max_warp_proportion));
				int base_right = base+int(minidtw_size*float(j-i)*(1+max_warp_proportion));
				if(subject_indices[j] <= base_right && subject_indices[j] >= base_left) {
					// membership[j] = i;
					membership[j] = member;
					count++;
				}
			}
			if(count > 0) {
				if(global_mem){
					// membership[i] = i + blockIdx.x*num_candidate_query_indices;;
					membership[i] = member + blockIdx.x*num_candidate_query_indices;;
				} else{
					// membership[i] = i;
					membership[i] = member;
				}
			}
			member++;
		}
	}
	__syncthreads();	
}

// Device side function that retrieves all distances from a membership and then sorts them in acending order
// sorted_colinear_distances - buffer that will store the sorted colinear distances
// membership - buffer containing the memberships of the indices
// distances - distances calculated from the dtw algorithm (hard/ soft)
// num_candidate_query_indices - the number of values to be stored in sorted_colinear_distances
// partition_counter - counter for which partition we are looking at
// local_partition - the currect partition we are on
// num_members - the number of members belonging to the membership that is being onbserved in a given thread
// num_sorted_colinear_distances - the total number of colinear distances found
// global_mem - flag that indicates if the two above buffers are global memory or shared memory
template <class T>
__inline__ __device__ void get_sorted_colinear_distances(QTYPE_ACC *sorted_colinear_distances, T* membership, QTYPE_ACC *distances, const int num_candidate_query_indices, int* partition_counter, int* local_partition, int num_members, int* num_sorted_colinear_distances, bool global_mem){
	
	// atomic add does not seem to be working, doing processes in order instead (see below)
	/*if(threadIdx.x > 0 && threadIdx.x < num_candidate_query_indices && num_members > 0) {
		local_partition = atomicAdd_block(partition_counter, num_members);
	}
	__syncthreads();*/
	for(int i = 0; i < num_candidate_query_indices; i++) {
		if(i == threadIdx.x && num_members > 0) {
			(*local_partition) = (*partition_counter);
			(*partition_counter) += num_members;
		}
		__syncthreads();
	}


	(*num_sorted_colinear_distances) = (*local_partition);
	QTYPE_ACC tmp;
	// Populate the set
	for(int i = threadIdx.x; i < num_candidate_query_indices && (*num_sorted_colinear_distances) < num_candidate_query_indices; i++){
		if(membership[i] == threadIdx.x){
			sorted_colinear_distances[(*num_sorted_colinear_distances)++] = distances[i];
		}
	}
	// Sort the set in place in L1
	for(int i = (*local_partition); i < (*local_partition)+num_members-1 && i < num_candidate_query_indices; i++){
		for(int j = i+1; j < (*local_partition)+num_members && j < num_candidate_query_indices; j++){
			if(sorted_colinear_distances[i] > sorted_colinear_distances[j]){
				tmp = sorted_colinear_distances[i];
				sorted_colinear_distances[i] = sorted_colinear_distances[j];
				sorted_colinear_distances[j] = tmp;
			}
		}
	}
}

// Device side function that retrieves all distances not belonging to a membership and then sorts them in acending order if the sorted_non_colinear_distances wasn't populated earlier
// sorted_non_colinear_distances - buffer that will store or currently stores the sorted non colinear distances
// membership - buffer containing the memberships of the indices
// num_members - the number of members in the current membership
// query_adjacent_distances - distances calculated from the dtw algorithm (hard/ soft)
// num_sorted_non_colinear_distances - the number of values that will be stored or are already stored in sorted_non_colinear_distances
// num_candidate_query_indices - number of candidate indices
// num_candidate_subject_indices - number of candidate subject indices
// thorough_calc - flag to check which pvalue function we are running
// global_non_colinear_distances - a global value for all sorted_non_colinear_distances
// num_sorted_colinear_distances - the total number of colinear distances found previously
// global_mem - flag that indicates if the two above buffers are global memory or shared memory
template <class T>
__inline__ __device__ void get_sorted_non_colinear_distances(QTYPE_ACC *sorted_non_colinear_distances, T* membership, int num_members, const QTYPE_ACC *query_adjacent_distances, int* num_sorted_non_colinear_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, bool thorough_calc, QTYPE_ACC *global_non_colinear_distances, int num_sorted_colinear_distances, bool global_mem){	

	if(thorough_calc){
		QTYPE_ACC tmp;
		(*num_sorted_non_colinear_distances) = 2*num_candidate_query_indices - num_sorted_colinear_distances;	// Because we want a size of 2 * num_candidate_query_indices for the Mann-Whitney test

		if((*num_sorted_non_colinear_distances) <= 0){	// In the case that we have a lot of colinear distances
			(*num_sorted_non_colinear_distances) = 0;
			return;
		}

		int num_iterations = num_members;

		int num_non_colinear_distances_per_query = (*num_sorted_non_colinear_distances) / num_iterations;	// Set the number of non colinear distances we want from each query that is part of the membership
		int remain_non_colinear_distances = (*num_sorted_non_colinear_distances) % num_iterations;	// Check how many more we need so we can grab extras in a few blocks if needed

		int remaining_dist_offset = 0; 
		if(remain_non_colinear_distances != 0){
			remaining_dist_offset = num_iterations / remain_non_colinear_distances;	// Offset to grab any remaning distances from
		}

		int cursor = 0;		// Position in sorted_non_colinear_distances
		int dist_block_position = threadIdx.x;	// Position in the block that we are looking at in future blocks
		bool membership_found = false;

		for(int cur_it = 0; cur_it < num_iterations; cur_it++){
			while(true){
				if(dist_block_position >= num_candidate_query_indices){
					dist_block_position = 0;
					break;
				}
				if(membership[dist_block_position] == threadIdx.x){
					membership_found = true;
					break;
				}
				dist_block_position++;
			}
			if(!membership_found){
				while(true){
					if(dist_block_position >= num_candidate_query_indices){
						printf("Error: A block position could not be found\n");
						(*num_sorted_non_colinear_distances) = 0;
						return;
					}
					if(membership[dist_block_position] != threadIdx.x){
						break;
					}
					dist_block_position++;
				}
			}
			
			int query_block_position = 1;	// Which query block in query_adjacent_distances are we currently in
			bool extra_dist_taken = false;
			// int values_taken_here = 0;
			while(true){
				// printf("%i * %i + %i = %i, %i * %i = %i\n", query_block_position, num_candidate_query_indices, dist_block_position, query_block_position * num_candidate_query_indices + dist_block_position, gridDim.x, blockDim.x, gridDim.x * blockDim.x);
				if(query_block_position * num_candidate_query_indices + dist_block_position >= gridDim.x*blockDim.x){ 
					break; // Check if we've gone past the bounds of query_adjacent_distances
				}
				// printf("query_adjacent_distances[%i * %i + %i]: %f\n", query_block_position, num_candidate_query_indices, dist_block_position, query_adjacent_distances[(query_block_position) * num_candidate_query_indices + dist_block_position]);
				sorted_non_colinear_distances[cursor++] = query_adjacent_distances[(query_block_position++) * num_candidate_query_indices + dist_block_position];
				if(extra_dist_taken){
					break;
				}

				if(cursor == (*num_sorted_non_colinear_distances)) break;	// Check if we've gotten enough values

				if(query_block_position > num_non_colinear_distances_per_query && 
				   remaining_dist_offset != 0 && 
				   cur_it % remaining_dist_offset != 0){
						break;	// Check if we need any extra values from the current set of distances
				}
				else if(query_block_position > num_non_colinear_distances_per_query && 
						remaining_dist_offset != 0 && 
						remain_non_colinear_distances == 0){
					break;
				}
				else if(query_block_position > num_non_colinear_distances_per_query && 
						remaining_dist_offset != 0 && 
						cur_it % remaining_dist_offset == 0 &&
						remain_non_colinear_distances != 0){
					remain_non_colinear_distances--;	// Reduce the count of remaining extra distances we need to grab if we got one from this set
					extra_dist_taken = true;	// Flag to let the next loop that we got the extra value from this set
				}
				else if(query_block_position > num_non_colinear_distances_per_query &&
						remaining_dist_offset == 0){ 
					break;	// Check if we've gotten enough values from the current set of distances
				}
			}
			if(cursor == (*num_sorted_non_colinear_distances)) break;
			dist_block_position++;
		}
		
		for(int k = 0; k < (*num_sorted_non_colinear_distances)-1; k++){
			for(int j = k; j < (*num_sorted_non_colinear_distances)-1; j++){
				if(sorted_non_colinear_distances[k] > sorted_non_colinear_distances[j]){
					tmp = sorted_non_colinear_distances[k]; 
					sorted_non_colinear_distances[k] = sorted_non_colinear_distances[j]; 
					sorted_non_colinear_distances[j] = tmp;
				}
			}
			// printf("sorted_non_colinear_distances[%i]: %f\n", k, sorted_non_colinear_distances[k]);
		}
	} else{
		int num_it = (*num_sorted_non_colinear_distances)/blockDim.x;
		int extra_it = (*num_sorted_non_colinear_distances)%blockDim.x;
		for(int it = 0; it < num_it; it++) {
			sorted_non_colinear_distances[blockDim.x*it+threadIdx.x] = global_non_colinear_distances[blockDim.x*it+threadIdx.x];
		}
		if(threadIdx.x < extra_it) {
			sorted_non_colinear_distances[blockDim.x*num_it+threadIdx.x] = global_non_colinear_distances[blockDim.x*num_it+threadIdx.x];
		}
		__syncthreads();
	}
}

// Device side function that calculates the pvalue of a membership from its colinear distances
// max_pval - the maximum pvalue used to accept or reject a match. If a pvalue is found to be lower than this value, then it wont be counted
// sorted_colinear_distances- colinear distances found for each set of memberships that are sorted in acending order stored based on tid
// sorted_non_colinear_distances - non colinear distances found that are sorted in acending order stored based on tid
// num_sorted_non_colinear_distances - the number of non colinear distances
// num_results_recorded - the total number of results that are found to be under the maximum pvalue by calculate_pval
// num_results_notrecorded - the total number of results that are found to be over the maximum by calculate_pval
// max_num_results - the maximum number of results that can be recorded
// output_pvals - the buffer that will store all pvalues based on its tid
// leftmost_anchor_query - the leftmost position in the query where the match has been found
// rightmost_anchor_query - the rightmost position in the query where the match has been found
// leftmost_anchor_subject - the leftmost position in the subject where the match has been found
// rightmost_anchor_subject - the rightmost position in the subject where the match has been found
// num_members - the number of members of the current membership. Is also the size of sorted_colinear_distances
// output_left_anchors_query - the buffer that will store all left anchors in the query based on its tid
// output_right_anchors_query - the buffer that will store all right anchors in the query based on its tid
// output_left_anchors_subject - the buffer that will store all left anchors in the subject based on its tid
// output_right_anchors_subject - the buffer that will store all right anchors in the subject based on its tid
// output_num_members - the buffer that will store the number of members that are looked at for each pvalue
// anch_mem_buff - he buffer that will store the number of members that are looked at for each pvalue based on the threads tid
// pval_buff - buffer used for testing that stores all pvalues based on its tid
// left_query_buff - the buffer used for testing that will store all left anchors in the query based on its tid. Wont be populated if it hasn't be allocated
// right_query_buff - the buffer used for testing that will store all right anchors in the query based on its tid. Wont be populated if it hasn't be allocated
// left_subject_buff - the buffer used for testing that will store all left anchors in the subject based on its tid. Wont be populated if it hasn't be allocated
// right_subject_buff - the buffer used for testing that will store all right anchors in the subject based on its tid. Wont be populated if it hasn't be allocated
__inline__ __device__ void calculate_pval(const float max_pval, QTYPE_ACC *sorted_colinear_distances, QTYPE_ACC *sorted_non_colinear_distances, int num_sorted_non_colinear_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int leftmost_anchor_query, int rightmost_anchor_query, long leftmost_anchor_subject, long rightmost_anchor_subject, int num_members, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff){

  // We must decide how many non-colinear values to include in the test (n) as this will change the final test statistic value.
  // One consistent option would be to always look at the same number of mini DTWs regardless of the number of colinear ones.
  // Since the co-linear ones max out at nTPB, and the best discrimination in the W test statistic is with an equal number in both 
  // categories, take as many sorted_non_colinear_distances as required to add up to 2*nTPB, if that many were provided.
  float return_pval = FLT_MAX;
  int m = num_members;
  int n = num_sorted_non_colinear_distances;

  int R = 0; // the test statistic
  int mi = 0, ni = 0;
  for(;mi < m; ++mi){
    while(ni < n && sorted_non_colinear_distances[ni] < sorted_colinear_distances[mi]){
      ++ni;
    }
    R += mi+ni;
  }
  if(m < 8 && m + n < 16){ // Edge case of query being of similar length to subject: the formula below for z-score isn't a good approximation in these cases
    int b = m*(m + n + 1); // upper-tail null hypothesis reject is w <= b - c (i.e. critical value)
    // lookup the critical values near 0.05, 0.025, 0.01 and 0.005, taken from Table A.10 in Jay L. Devore (1995) ISBN 0-534-24264-2
    if(m == 3){
      switch(n){
        case 0:
        case 1:
        case 2: return_pval = 1; break;
        case 3: return_pval = (R <= b - 15 ? 0.05 : 1); break;
        case 4: return_pval = (R <= b - 18 ? 0.029 : (R <= b - 17 ? 0.057 : 1)); break;
        case 5: return_pval = (R <= b - 21 ? 0.018 : (R <= b - 20 ? 0.036 : 1)); break;
        case 6: return_pval = (R <= b - 24 ? 0.012 : (R <= b - 23 ? 0.024 : (R <= b - 22 ? 0.048 : 1))); break;
        case 7: return_pval = (R <= b - 27 ? 0.008 : (R <= b - 26 ? 0.017 : (R <= b - 24 ? 0.058 : 1))); break;
        default: return_pval = (R <= b - 30 ? 0.006 : (R <= b - 29 ? 0.012 : (R <= b - 28 ? 0.024 : (R <= b - 27 ? 0.042 : 1)))); // will be overly cautious for m = 3 and 13 > n > 8
      }
    }
    else if(m == 4){
      switch(n){
        case 0:
        case 1:
        case 2:
        case 3: return_pval = 1; break;
        case 4: return_pval = (R <= b - 26 ? 0.014 : (R <= b - 25 ? 0.029 : (R <= b - 24 ? 0.057 : 1))); break;
        case 5: return_pval = (R <= b - 30 ? 0.008 : (R <= b - 29 ? 0.016 : (R <= b - 28 ? 0.032 : (R <= b - 27 ? 0.056 : 1)))); break;
        case 6: return_pval = (R <= b - 34 ? 0.005 : (R <= b - 33 ? 0.010 : (R <= b - 32 ? 0.019 : (R <= b - 30 ? 0.057 : 1)))); break;
        case 7: return_pval = (R <= b - 37 ? 0.006 : (R <= b - 36 ? 0.012 : (R <= b - 35 ? 0.021 : (R <= b - 33 ? 0.055 : 1)))); break;
        default: return_pval = (R <= b - 41 ? 0.004 : (R <= b - 40 ? 0.008 : (R <= b - 38 ? 0.024 : (R <= b - 36 ? 0.055 : 1)))); // will be overly cautious for m = 4 and 12 > n > 8
      }
    }
    else if(m == 5){
      switch(n){
        case 0:
        case 1:
        case 2:
        case 3:
        case 4: return_pval = 1; break;
        case 5: return_pval = (R <= b - 40 ? 0.004 : (R <= b - 39 ? 0.008 : (R <= b - 37 ? 0.028 : (R <= b - 36 ? 0.048 : 1)))); break;
        case 6: return_pval = (R <= b - 44 ? 0.004 : (R <= b - 43 ? 0.009 : (R <= b - 41 ? 0.026 : (R <= b - 40 ? 0.041 : 1)))); break;
        case 7: return_pval = (R <= b - 48 ? 0.005 : (R <= b - 47 ? 0.009 : (R <= b - 45 ? 0.024 : (R <= b - 43 ? 0.053 : 1)))); break;
        default: return_pval = (R <= b - 52 ? 0.005 : (R <= b - 51 ? 0.009 : (R <= b - 49 ? 0.023 : (R <= b - 47 ? 0.047 : 1)))); // will be overly cautious for m = 5 and 11 > n > 8
      }
    }
    else if(m == 6){
      switch(n){
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5: return_pval = 1; break;
		case 6: return_pval = (R <= b - 55 ? 0.004 : (R <= b - 54 ? 0.008 : (R <= b - 52 ? 0.021 : (R <= b - 50 ? 0.047 : 1)))); break;
        case 7: return_pval = (R <= b - 60 ? 0.004 : (R <= b - 58 ? 0.011 : (R <= b - 56 ? 0.026 : (R <= b - 54 ? 0.051 : 1)))); break;
        default: return_pval = (R <= b - 52 ? 0.005 : (R <= b - 51 ? 0.009 : (R <= b - 49 ? 0.023 : (R <= b - 47 ? 0.047 : 1)))); // will be overly cautious for m = 6 and n = 9
      }
    }
    else if(m == 7){
      switch(n){
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6: return_pval = 1; break;
        case 7: return_pval = (R <= b - 72 ? 0.006 : (R <= b - 71 ? 0.009 : (R <= b - 68 ? 0.027 : (R <= b - 66 ? 0.049 : 1)))); break;
        default: return_pval = (R <= b - 78 ? 0.005 : (R <= b - 76 ? 0.01 : (R <= b - 73 ? 0.027 : (R <= b - 71 ? 0.047 : 1)))); // a.k.a. case 8 because of enclosing m + n < 16 condition
      }
    }
    else{ // have enough co-linear samples and so few non-colinear that these tests fail and we can assume it's not random at all
      return_pval = 0;
    }
  }
  else{
    // Assumption: Note that as these are continuous DTW distance values, we expect very few ties unless there is zero noise in the system, and so don't worry about their effect on the W statistic
    double z = ((R - (m*(m+1)/2.0))-(m*n/2.0))/sqrt(m*n*(m+n+1)/12.0);
    // if(z < -6){
      // r = 0.0; // Underflow in actual calc will cause bad result if expanded calculation is done, based on previous experience.
    // }
    // else if(z > 6){ // Same underflow deal.
      // r = 1.0;
    // }
    // else{
        // double value = z, sum = z;
        // for(int i=1; i < 1000; ++i){ // 100 gives us more than enough precision
            // value *= z*z/(2*i+1);
            // sum += value;
        // }
        // return_pval = 0.5+sum*0.3989*exp(-(z*z)/2.0); // z-score -> p-value estimation formula
        return_pval = 0.5*(erf(z/pow(2.0, 0.5))+1); // z-score -> p-value estimation formula
		// printf("R: %f, m: %i, n: %i, z: %f, return_pval: %f\n", R, m, n, z, return_pval);
    // }
  }

  if(return_pval < max_pval){ // meets threshold for inclusion

    // Write the results of the co-linear match back to global memory, which requires some coordination. 
    // This increment is atomic with respect to the GPU device only (not the CPU or other GPUs).
    int my_result_ordinal = atomicInc(num_results_recorded, max_num_results);
    // This means that we've exceeded max_num_results. Find a less-good result already recorded (if any) and kick it out.
    if(my_result_ordinal == max_num_results){ 
      // Keep track of the total number of matches even if we don't record the details of them all. This will allow us to estimate a suitable
      // multiple testing correction later to the top max_num_results results.
      atomicInc(num_results_notrecorded, INT_MAX);
    }
    else{
      if(output_num_members != 0) output_num_members[my_result_ordinal] = num_members; // These can be used as anchors for a quilted final DTW if the caller wanted a full DTW
      output_pvals[my_result_ordinal] = return_pval;
      output_left_anchors_query[my_result_ordinal] = leftmost_anchor_query;
      output_right_anchors_query[my_result_ordinal] = rightmost_anchor_query;
      output_left_anchors_subject[my_result_ordinal] = leftmost_anchor_subject;
      output_right_anchors_subject[my_result_ordinal] = rightmost_anchor_subject;
	  int tid = blockIdx.x*blockDim.x+threadIdx.x;
      if(anch_mem_buff != 0) anch_mem_buff[tid] = num_members; // These can be used as anchors for a quilted final DTW if the caller wanted a full DTW
      if(pval_buff != 0){ // Don't populate buffers if we aren't runing our tests
        pval_buff[tid] = return_pval;
        left_query_buff[tid] = leftmost_anchor_query;
        right_query_buff[tid] = rightmost_anchor_query;
        left_subject_buff[tid] = leftmost_anchor_subject;
        right_subject_buff[tid] = rightmost_anchor_subject;
      }
    }
  }
}

// Function that finds the longest increasing subsequence in a given membership and then removes all positions in the membership that are not part of that subsequence
// Code obtained and modified based on https://www.geeksforgeeks.org/c-program-for-longest-increasing-subsequence/
// subject_indices - Positions in the subject where a match was found
// membership - the memberships belonging to the indices
// lis - buffer to contain the length of the longest increasing subsequence
// indices - buffer to contain temp indices of each longest increasing subsequence found
// longest_indices - buffer to contain the indices of the true longest increasing subsequence
// num_candidate_query_indices - the number of values contained in subject_indices and membership
template <class T>
__inline__ __device__ void longestIncreasingSubsequence(long long *subject_indices, T* membership, int* lis, int* lis_indices, int* longest_indices, const int num_candidate_query_indices){

	// Initialize all values in lis and lis_indices
	for (int i = threadIdx.x; i < num_candidate_query_indices; i++){
		if(membership[i] == threadIdx.x){
			lis[i] = 1;
			lis_indices[i] = -1;
		}
	}

	lis_indices[threadIdx.x] = threadIdx.x;
	int largest = 1;
	// Creation of the longest increasing subsequence
	for (int i = threadIdx.x+1; i < num_candidate_query_indices; i++) {
		for (int j = threadIdx.x; j < i; j++) {
			// Check if the memberships belong to the membeship we want to compare
			if (subject_indices[i] > subject_indices[j] && lis[i] < lis[j] + 1 && membership[i] == threadIdx.x && membership[j] == threadIdx.x){ 
				lis[i] = lis[j] + 1;
				lis_indices[j] = j;	// Assign the index where a possible longest increasing subsequence was found
			}
		}
		if(membership[i] == threadIdx.x) lis_indices[i] = i;
		// If the current longest increasing subsequence is larger than one that we have found before, store the indices of that subsequence in the longest_indices buffer
        if(lis[i] > largest && membership[i] == threadIdx.x){
        	largest = lis[i];
          	for(int k = threadIdx.x; k < num_candidate_query_indices; k++){
				if(membership[k] == threadIdx.x){
					longest_indices[k] = lis_indices[k];
					// Reset the lis_indices buffer
					lis_indices[k] = -1;
				}
        	}
        }
	}
	// Remove all memberships that are nor part of the longest increasing subsequence
	if(largest > 1){
		for(int i = threadIdx.x; i < num_candidate_query_indices; i++){
			if(longest_indices[i] == -1 && membership[i] == threadIdx.x){
				membership[i] = -1;
			}
		}
	}
}

// Overlap the data in the threadblocks by 100 candidate_subject_indices, because internally we can only definitely detect colinearity of two blocks
// if they are within 95 query blocks of each other. The threadblock data overlap means we won't miss any that straddle off the end of a given threadblock's data bailiwick
// because it will be fully contained in an adjacent threadblock.

// Note that we need to be really stingy with the local variables declared within this kernel, because if we have CUDA_THREADBLOCK_MAX_THREADS threads per block at runtime, we only get 
// ~48 local variables (we only ask for 14338 in shared memory if CUDA_THREADBLOCK_MAX_THREADS threads so CUDA runtime will automagically pick the 16K/48K config where 48K is L1 cache for local variables and 16K is for shared memory)
// and with more local variables you'll get a "too many resources requested for launch." runtime CUDA error.

__global__
void
thorough_calc_anchor_candidates_colinear_pvals(const long long *query_adjacent_candidates, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, const int expected_num_pvals, const float max_warp_proportion, const float max_pval, const QTYPE_ACC *query_adjacent_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, const int minidtw_size, int* all_memberships, long long* all_subject_indices, int* lis_memberships, QTYPE_ACC* colinear_buff, QTYPE_ACC* non_colinear_buff, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff){
	if(blockIdx.x == 0 && threadIdx.x == 0){
		// Make sure num_results_recorded and num_results_notrecorded are intialized
		atomicExch(num_results_recorded, 0);
		atomicExch(num_results_notrecorded, 0);
	}
	
	extern __shared__ char shared[];
	long long *subject_indices = (long long *) &shared[0];
	QTYPE_ACC *distances = (QTYPE_ACC *) &subject_indices[num_candidate_query_indices];
	QTYPE_ACC *sorted_colinear_distances = (QTYPE_ACC *) &distances[num_candidate_query_indices];
	int* lis = (int *) &sorted_colinear_distances[num_candidate_query_indices];
	int* lis_indices = (int *) &lis[num_candidate_query_indices];
	int* longest_indices = (int *) &lis_indices[num_candidate_query_indices];
	short *membership = (short *) &longest_indices[num_candidate_query_indices];

	if(threadIdx.x < num_candidate_query_indices) {
		subject_indices[threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		distances[threadIdx.x] = query_adjacent_distances[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		membership[threadIdx.x] = -1;
		#if defined(_FLASH_UNIT_TEST)
		all_memberships[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		all_subject_indices[blockIdx.x*blockDim.x+threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		colinear_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		non_colinear_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		anch_mem_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		left_query_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		right_query_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		left_subject_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		right_subject_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		#endif
	}
	__syncthreads();

	set_membership_results<short>(membership, num_candidate_query_indices, subject_indices, minidtw_size, max_warp_proportion, false);
	
	// set_membership_results
	// int member = 0;
	// if(threadIdx.x == 0) {
		// for(int i = 0; i < num_candidate_query_indices; i++) {
			// if(membership[i] != -1) {
				// continue;
			// }
			// long long base = subject_indices[i];
			// int count = 0;
			// for(int j = i+1; j < num_candidate_query_indices; j++) {
				// if(membership[j] != -1)
					// continue;
				// int base_left = base+int(minidtw_size*float(j-i)*(1-max_warp_proportion));
				// int base_right = base+int(minidtw_size*float(j-i)*(1+max_warp_proportion));
				// if(subject_indices[j] <= base_right && subject_indices[j] >= base_left) {
					// membership[j] = member;
					// count++;
				// }
			// }
			// if(count > 0) {
				// membership[i] = member;
				// member++;
			// }
		// }
	// }
	// __syncthreads();
	// end of set_membership_results
	#if defined(_FLASH_UNIT_TEST)
	if(threadIdx.x == 0){
		for(int i = 0; i < num_candidate_query_indices; i++){
			all_memberships[blockIdx.x*num_candidate_query_indices+i] = membership[i] == -1 ? membership[i] : membership[i] + blockIdx.x*num_candidate_query_indices;
		}
	}
	#endif

	if(membership[threadIdx.x] == threadIdx.x){
		longestIncreasingSubsequence<short>(subject_indices, membership, lis, lis_indices, longest_indices, num_candidate_query_indices-threadIdx.x);		
	}
	__syncthreads();

	#if defined(_FLASH_UNIT_TEST)
	if(threadIdx.x == 0){
		for(int i = 0; i < num_candidate_query_indices; i++){
			lis_memberships[blockIdx.x*num_candidate_query_indices+i] = membership[i] == -1 ? membership[i] : membership[i] + blockIdx.x*num_candidate_query_indices;
		}
	}
	#endif

	int leftmost_anchor_query = -1;
	int rightmost_anchor_query = -1;
	int leftmost_anchor_subject = -1;
	int rightmost_anchor_subject = -1;
	int num_members = 0;
	for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){ // start at threadIdx.x because we know that the leftmost member of the disjoint set being examined is the thread being executed
		if(membership[i] == threadIdx.x){
			if(leftmost_anchor_query == -1){
				leftmost_anchor_query = i*minidtw_size;
				leftmost_anchor_subject = subject_indices[i];
			}
			rightmost_anchor_query = i*minidtw_size;
			rightmost_anchor_subject = subject_indices[i];
			num_members++;
		}
	}
	__syncthreads();
	
	// int *partition_counter = (int *) &membership[num_candidate_query_indices+(num_candidate_query_indices%2)];
	int *partition_counter = (int *) &lis[0];
	if(threadIdx.x == 0) {
		(*partition_counter) = 0;
	}
	__syncthreads();

	if(num_members == 0){
		return;
	}

	int local_partition = 0;
	int num_sorted_colinear_distances = 0;

	get_sorted_colinear_distances<short>(sorted_colinear_distances, membership, distances, num_candidate_query_indices, partition_counter, &local_partition, num_members, &num_sorted_colinear_distances, false);
	
	#if defined(_FLASH_UNIT_TEST)
	if(threadIdx.x == 0){
		for(int i = 0; i < num_sorted_colinear_distances; i++){
			colinear_buff[blockIdx.x*num_candidate_query_indices+i] = sorted_colinear_distances[i];
		}
	}
	#endif
	
	// if(threadIdx.x == 0) {
		// *partition_counter = 0;
	// }
	// __syncthreads();
	
	// // atomic add does not seem to be working, doing processes in order instead (see below)
	// /*if(threadIdx.x > 0 && threadIdx.x < num_candidate_query_indices && num_members > 0) {
		// local_partition = atomicAdd_block(partition_counter, num_members);
	// }
	// __syncthreads();*/
	// for(int i = 0; i < num_candidate_query_indices; i++) {
		// if(i == threadIdx.x && num_members > 0) {
			// local_partition = *partition_counter;
			// *partition_counter += num_members;
		// }
		// __syncthreads();
	// }
	
	// int cursor = local_partition;
	// QTYPE_ACC tmp;
	// Populate the set

	// get_sorted_colinear_distances
	// for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){
		// if(membership[i] == threadIdx.x){
			// sorted_colinear_distances[cursor++] = distances[i];
		// }
	// }
	// // Sort the set in place in L1
	// for(int i = local_partition; i < local_partition+num_members-1; i++){
		// for(int j = local_partition+1; j < local_partition+num_members; j++){
			// if(sorted_colinear_distances[i] > sorted_colinear_distances[j]){
				// tmp = sorted_colinear_distances[i]; sorted_colinear_distances[i] = sorted_colinear_distances[j]; sorted_colinear_distances[j] = tmp;
			// }
		// }
	// }

	// end of sorted_colinear_distances
	
	QTYPE_ACC *sorted_non_colinear_distances = (QTYPE_ACC *) &global_non_colinear_distances[CUDA_THREADBLOCK_MAX_THREADS*(threadIdx.x%expected_num_pvals)+CUDA_THREADBLOCK_MAX_THREADS*expected_num_pvals*blockIdx.x];
	
	// TODO needs testing for small num_candidate_subject_indices
	int num_sorted_non_colinear_distances = 0;
	// for(int i = 0; i < num_candidate_query_indices; i += expected_num_pvals) {
		// if(i+expected_num_pvals > threadIdx.x && i <= threadIdx.x && num_members > 0) {
			get_sorted_non_colinear_distances<short>(sorted_non_colinear_distances, membership, num_members, query_adjacent_distances, &num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, true, 0, num_sorted_colinear_distances, false);
			
			// TODO: Figure out why this is broken
			// #if defined(_FLASH_UNIT_TEST)
				// if(threadIdx.x == 0){
					// for(int i = 0; i < num_sorted_non_colinear_distances; i++){
						// non_colinear_buff[blockIdx.x*num_candidate_query_indices+i] = sorted_non_colinear_distances[i];
					// }
				// }
			// #endif
			// get_sorted_non_colinear_distances
			// num_sorted_non_colinear_distances = CUDA_THREADBLOCK_MAX_THREADS;
			// if(num_candidate_query_indices*num_candidate_subject_indices < num_sorted_non_colinear_distances)
				// num_sorted_non_colinear_distances = num_candidate_query_indices*num_candidate_subject_indices;
			// int num_iterations = num_members;
			// int num_samples_per_it = num_sorted_non_colinear_distances/num_members;
			// int num_bonus_it = num_sorted_non_colinear_distances%num_members;
			// int extra_it_count = 0;
			// int step_size = num_candidate_subject_indices/(num_samples_per_it+(num_bonus_it ? 1 : 0)+1);
			// if(num_samples_per_it+(num_bonus_it ? 1 : 0)+1 > num_candidate_subject_indices) {
				// num_samples_per_it = num_candidate_subject_indices;
				// num_iterations = num_sorted_non_colinear_distances/num_samples_per_it;
				// num_bonus_it = 0;
				// extra_it_count = num_sorted_non_colinear_distances%num_samples_per_it;
				// step_size = 1;
			// }
			// cursor = 0;
			// int k = threadIdx.x; // should follow members while num_members < num_iterations and then fill with -1 elements
			// bool oob_switch = false; // I know oob is a wierd name, stands for out-of-bounds
			// // iterate through members, choosing (if possible) from elements not included in colinear candidates
			// // choose from member ids starting with those within the pval calculation, and then those not in the pval calculation
			// for(int it = 0; it < num_iterations; it++) {
				// if(!oob_switch) {
					// while(true) {
						// if(k >= num_candidate_query_indices) {
							// k = 0;
							// oob_switch = true;
							// break;
						// }
						// if(membership[k] == threadIdx.x)
							// break;
						// k++;
					// }
				// }
				// if(oob_switch) {
					// while(true) {
						// if(k >= num_candidate_query_indices) printf("error: bid %i, tid %i, it %i, nit %i\n", blockIdx.x, threadIdx.x, it, num_iterations);
						// if(membership[k] != threadIdx.x)
							// break;
						// k++;
					// }
				// }
				// for(int j = 0; j < num_samples_per_it+(it < num_bonus_it ? 1 : 0); j++) {
					// if(j*step_size == blockIdx.x)
						// j++;
					// if(j*step_size < num_candidate_subject_indices) {
						// sorted_non_colinear_distances[cursor++] = query_adjacent_distances[k+(j)*step_size*num_candidate_query_indices];
					// } else if(step_size > 1) {
						// sorted_non_colinear_distances[cursor++] = query_adjacent_distances[k+(blockIdx.x+1)*num_candidate_query_indices];
					// } else {
						// sorted_non_colinear_distances[cursor++] = query_adjacent_distances[k+(blockIdx.x)*num_candidate_query_indices];
					// }
				// }
				// k++;
			// }
			// if(extra_it_count) {
				// if(!oob_switch) {
					// while(true) {
						// if(k >= num_candidate_query_indices) {
							// k = 0;
							// oob_switch = true;
							// break;
						// }
						// if(membership[k] == threadIdx.x)
							// break;
						// k++;
					// }
				// }
				// if(oob_switch) {
					// while(true) {
						// if(membership[k] != threadIdx.x)
							// break;
						// k++;
					// }
				// }
				// for(int j = 0; j < extra_it_count; k++) {
					// // no need for j*step_size < num_candidate_subject_indices check, in order to get here extra_it_count < num_samples_per_it
					// if(j*step_size == blockIdx.x)
						// j++;
					// sorted_non_colinear_distances[cursor++] = query_adjacent_distances[k+(j)*step_size*num_candidate_query_indices];
				// }
			// }
			
			// for(int k = 0; k < num_sorted_non_colinear_distances-1; k++){
				// for(int j = 1; j < num_sorted_non_colinear_distances; j++){
					// if(sorted_non_colinear_distances[k] > sorted_non_colinear_distances[j]){
						// tmp = sorted_non_colinear_distances[k]; sorted_non_colinear_distances[k] = sorted_non_colinear_distances[j]; sorted_non_colinear_distances[j] = tmp;
					// }
				// }
			// }
			// end of sorted_non_colinear_distances
			// TODO got here, above is taking too long, speed it up!
			
			calculate_pval(
				max_pval,
				&sorted_colinear_distances[local_partition],
				sorted_non_colinear_distances,
				num_sorted_non_colinear_distances,
				num_results_recorded,
				num_results_notrecorded,
				max_num_results,
				output_pvals,
				leftmost_anchor_query,
				rightmost_anchor_query,
				leftmost_anchor_subject,
				rightmost_anchor_subject,
				num_members,
				output_left_anchors_query,
				output_right_anchors_query,
				output_left_anchors_subject,
				output_right_anchors_subject,
				output_num_members,
				anch_mem_buff,
				pval_buff,
				left_query_buff,
				right_query_buff,
				left_subject_buff,
				right_subject_buff
			);
		// }
		// __syncthreads();
	// }
}

__global__
void
fast_calc_anchor_candidates_colinear_pvals(const long long *query_adjacent_candidates, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, int *num_non_colinear_distances, const float max_warp_proportion, const float max_pval, const QTYPE_ACC *query_adjacent_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, int *output_num_members, const int minidtw_size, int* all_memberships, long long* all_subject_indices, int* lis_memberships, QTYPE_ACC* colinear_buff, QTYPE_ACC* non_colinear_buff, int* anch_mem_buff, float* pval_buff, int* left_query_buff, int* right_query_buff, long* left_subject_buff, long* right_subject_buff){
	if(blockIdx.x == 0 && threadIdx.x == 0){
		// Make sure num_results_recorded and num_results_notrecorded are intialized
		atomicExch(num_results_recorded, 0);
		atomicExch(num_results_notrecorded, 0);
	}
	
	extern __shared__ char shared[];
	long long *subject_indices = (long long *) &shared[0];
	QTYPE_ACC *distances = (QTYPE_ACC *) &subject_indices[num_candidate_query_indices];
	QTYPE_ACC *sorted_colinear_distances = (QTYPE_ACC *) &distances[num_candidate_query_indices];
	short *membership = (short *) &sorted_colinear_distances[num_candidate_query_indices];
	__shared__ QTYPE_ACC sorted_non_colinear_distances[CUDA_THREADBLOCK_MAX_THREADS];
	int* lis = (int *) &sorted_non_colinear_distances[num_candidate_query_indices];
	int* lis_indices = (int *) &lis[num_candidate_query_indices];
	int* longest_indices = (int *) &lis_indices[num_candidate_query_indices];

	if(threadIdx.x < num_candidate_query_indices) {
		subject_indices[threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		all_subject_indices[blockIdx.x*blockDim.x+threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		distances[threadIdx.x] = query_adjacent_distances[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		membership[threadIdx.x] = -1;
		all_memberships[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		colinear_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
		non_colinear_buff[blockIdx.x*num_candidate_query_indices+threadIdx.x] = -1;
	}
	__syncthreads();
	
	set_membership_results<short>(membership, num_candidate_query_indices, subject_indices, minidtw_size, max_warp_proportion, false);

	// set_membership_results
	// int member = 0;
	// if(threadIdx.x == 0) {
		// for(int i = 0; i < num_candidate_query_indices; i++) {
			// if(membership[i] != -1) {
				// continue;
			// }
			// long long base = subject_indices[i];
			// int count = 0;
			// for(int j = i+1; j < num_candidate_query_indices; j++) {
				// if(membership[j] != -1)
					// continue;
				// int base_left = base+int(minidtw_size*float(j-i)*(1-max_warp_proportion));
				// int base_right = base+int(minidtw_size*float(j-i)*(1+max_warp_proportion));
				// if(subject_indices[j] <= base_right && subject_indices[j] >= base_left) {
					// membership[j] = member;
					// count++;
				// }
			// }
			// if(count > 0) {
				// membership[i] = member;
				// member++;
			// }
		// }
	// }
	// __syncthreads();
	// end membership

	if(threadIdx.x == 0){
		for(int i = 0; i < num_candidate_query_indices; i++){
			all_memberships[blockIdx.x*num_candidate_query_indices+i] = membership[i] == -1 ? membership[i] : membership[i] + blockIdx.x*num_candidate_query_indices;
		}
	}

	if(membership[threadIdx.x] == threadIdx.x){
		longestIncreasingSubsequence<short>(subject_indices, membership, lis, lis_indices, longest_indices, num_candidate_query_indices-threadIdx.x);		
	}
	__syncthreads();
	if(threadIdx.x == 0){
		for(int i = 0; i < num_candidate_query_indices; i++){
			lis_memberships[blockIdx.x*num_candidate_query_indices+i] = membership[i] == -1 ? membership[i] : membership[i] + blockIdx.x*num_candidate_query_indices;
		}
	}
	
	int leftmost_anchor_query = -1;
	int rightmost_anchor_query = -1;
	int leftmost_anchor_subject = -1;
	int rightmost_anchor_subject = -1;
	int num_members = 0;
	for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){ // start at threadIdx.x because we know that the leftmost member of the disjoint set being examined is the thread being executed
		if(membership[i] == threadIdx.x){
			if(leftmost_anchor_query == -1){
				leftmost_anchor_query = i*minidtw_size;
				leftmost_anchor_subject = subject_indices[i];
			}
			rightmost_anchor_query = i*minidtw_size;
			rightmost_anchor_subject = subject_indices[i];
			num_members++;
		}
	}
	__syncthreads();
	
	int *partition_counter = (int *) &membership[num_candidate_query_indices+(num_candidate_query_indices%2)];
	int local_partition = 0;
	int num_sorted_colinear_distances = 0;

	get_sorted_colinear_distances<short>(sorted_colinear_distances, membership, distances, num_candidate_query_indices, partition_counter, &local_partition, num_members, &num_sorted_colinear_distances, false);
	
	// if(threadIdx.x == 0) {
		// *partition_counter = 0;
	// }
	// __syncthreads();
	
	// // atomic add does not seem to be working, doing processes in order instead (see below)
	// /*if(threadIdx.x > 0 && threadIdx.x < num_candidate_query_indices && num_members > 0) {
		// local_partition = atomicAdd_block(partition_counter, num_members);
	// }
	// __syncthreads();*/
	// for(int i = 0; i < num_candidate_query_indices; i++) {
		// if(i == threadIdx.x && num_members > 0) {
			// local_partition = *partition_counter;
			// *partition_counter += num_members;
		// }
		// __syncthreads();
	// }
	
	// // start get_sorted_colinear_distances
	// int cursor = local_partition;
	// QTYPE_ACC tmp;
	// // Populate the set
	// for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){
		// if(membership[i] == threadIdx.x){
			// sorted_colinear_distances[cursor++] = distances[i];
		// }
	// }
	// // Sort the set in place in L1
	// for(int i = local_partition; i < local_partition+num_members-1; i++){
		// for(int j = local_partition+1; j < local_partition+num_members; j++){
			// if(sorted_colinear_distances[i] > sorted_colinear_distances[j]){
				// tmp = sorted_colinear_distances[i]; sorted_colinear_distances[i] = sorted_colinear_distances[j]; sorted_colinear_distances[j] = tmp;
			// }
		// }
	// }
	// end sorted_colinear_distances
	// start get_sorted_non_colinear_distances
	get_sorted_non_colinear_distances<short>(sorted_non_colinear_distances, membership, num_members, query_adjacent_distances, num_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, false, global_non_colinear_distances, num_sorted_colinear_distances, false);
	// int num_it = (*num_non_colinear_distances)/blockDim.x;
	// int extra_it = (*num_non_colinear_distances)%blockDim.x;
	// for(int it = 0; it < num_it; it++) {
		// sorted_non_colinear_distances[blockDim.x*it+threadIdx.x] = global_non_colinear_distances[blockDim.x*it+threadIdx.x];
	// }
	// if(threadIdx.x < extra_it) {
		// sorted_non_colinear_distances[blockDim.x*num_it+threadIdx.x] = global_non_colinear_distances[blockDim.x*num_it+threadIdx.x];
	// }
	// __syncthreads();
	// end sorted_non_colinear_distances
	
	calculate_pval(
		max_pval,
		&sorted_colinear_distances[local_partition],
		sorted_non_colinear_distances,
		*num_non_colinear_distances,
		num_results_recorded,
		num_results_notrecorded,
		max_num_results,
		output_pvals,
		leftmost_anchor_query,
		rightmost_anchor_query,
		leftmost_anchor_subject,
		rightmost_anchor_subject,
		num_members,
		output_left_anchors_query,
		output_right_anchors_query,
		output_left_anchors_subject,
		output_right_anchors_subject,
		output_num_members,
		anch_mem_buff,
		pval_buff,
		left_query_buff,
		right_query_buff,
		left_subject_buff,
		right_subject_buff
	);
}

__global__
void
load_non_colinear_distances(const QTYPE_ACC *query_adjacent_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, int *num_non_colinear_distances) {
	__shared__ int partition_counter[1];
	
	if(threadIdx.x == 0)
		*num_non_colinear_distances = CUDA_THREADBLOCK_MAX_THREADS;
	
	int samples_per_query = CUDA_THREADBLOCK_MAX_THREADS/num_candidate_query_indices;
	int extra_samples = CUDA_THREADBLOCK_MAX_THREADS%num_candidate_query_indices;
	int num_samples = samples_per_query+(threadIdx.x < extra_samples ? 1 : 0);
	if(samples_per_query+(extra_samples ? 1 : 0) > num_candidate_subject_indices) {
		samples_per_query = num_candidate_subject_indices;
		extra_samples = 0;
		num_samples = samples_per_query;
		if(threadIdx.x == 0)
			*num_non_colinear_distances = samples_per_query*num_candidate_query_indices;
	}
	
	if(threadIdx.x == 0)
		*partition_counter = 0;
	__syncthreads();
	
	int partition = 0;
	for(int i = 0; i < num_candidate_query_indices; i++) {
		if(threadIdx.x == i) {
			partition = *partition_counter;
			*partition_counter += num_samples;
		}
		__syncthreads();
	}
	
	int sample_spacing = num_candidate_subject_indices/samples_per_query;
	for(int i = 0; i < num_samples; i++) {
		global_non_colinear_distances[partition+i] = query_adjacent_distances[threadIdx.x+i*sample_spacing*num_candidate_query_indices];
	}
}
		
// Finds the minimum value in a warp (here of 32) and returns the result
// val - the values in the warp that will be compared with one another
// returns the value found to be the minimum in the warp
__inline__ __device__ int warpReduceMin(int val){
    for (int offset = CUDA_WARP_WIDTH/ 2; offset > 0; offset /= 2){
        int tmpVal = __shfl_down_sync(FULL_MASK, val, offset);
        if (tmpVal < val){
            val = tmpVal;
        }
    }
    return val;
}

// We have an input array consisting of a bunch of CUDA_THREADBLOCK_MAX_THREADS sized sorted pvalues, and need to rank them up to max_rank (or all if max_rank < 1).
// Trying to minimize writes to global memory here.
__global__
void
merge_ranks(float *anchor_pvals, int num_pvals, int max_rank, int *anchor_pval_ranks, unsigned int *anchor_num_ranks){
    extern __shared__ int warp_minvals[];
    int *threadblock_minval = warp_minvals + sizeof(int)*CUDA_WARP_WIDTH;
    unsigned int *num_blocks_finished = (unsigned int *) (threadblock_minval + sizeof(int));
    *num_blocks_finished = 0;
    int block_cursor = threadIdx.x*CUDA_THREADBLOCK_MAX_THREADS;
    
    int lane = threadIdx.x % CUDA_WARP_WIDTH;
    int wid = threadIdx.x / CUDA_WARP_WIDTH;

    // Resolve the ranks within each thread warp.
    unsigned int global_rank = 1;
    while(*num_blocks_finished < blockDim.x){
      float minval = block_cursor >= 0 ?
                        anchor_pvals[block_cursor] : FLT_MAX; // these reads will be SO not coalesced
      minval = warpReduceMin(minval);
      if(!lane) warp_minvals[wid] = minval;
      __syncthreads();  
      // Get in-bounds values only for final threadblock reduction (threadblock may not be full).
      minval = (threadIdx.x < blockDim.x / CUDA_WARP_WIDTH) ? warp_minvals[lane] : FLT_MAX;
      if(!wid) *threadblock_minval = warpReduceMin(minval);
      __syncthreads();  

      // Resolve the ranks globally, now that we know the lowest value not processed so far.
      minval = *threadblock_minval;
      while(anchor_pvals[block_cursor] == minval){
        anchor_pval_ranks[block_cursor] = global_rank;
        block_cursor++;
        // Nothing more to process in this block?
        if(block_cursor >= num_pvals || block_cursor/CUDA_THREADBLOCK_MAX_THREADS != threadIdx.x/CUDA_THREADBLOCK_MAX_THREADS){
          block_cursor = -1; // effectively ends processing by this thread
          atomicInc(num_blocks_finished, INT_MAX);
		  break;
        }
      }
      global_rank++;
      if(max_rank > 0 && global_rank > max_rank){
        // End the processing in every thread, setting remaining rank to zero (i.e. ignore)
        while(block_cursor < num_pvals && block_cursor/CUDA_THREADBLOCK_MAX_THREADS == threadIdx.x/CUDA_THREADBLOCK_MAX_THREADS){
          anchor_pval_ranks[block_cursor++] = 0;
        }
        break;
      }
    }

    *anchor_num_ranks = global_rank-1;
}

// Rank within each threadblock, we are trying to move the least amount of data possible in this kernel, so we sort a local copy.
__global__
void
rank_results_blocks(float *input_pvals, int input_pvals_length, int *sorted_pval_ordinals, int *sorted_pval_ranks){
    // The FDR is based on the rank of the p-value in the sorted observation list. We don't assume a sorted input list.
    // To maintain the order of the pvals when converted to qvals, we need to make a copy of the pvals and
    // co-sort them with their ordinal value to determine the rank (including even rank for ties) of each input_pval. 
    // Do everything in shared memory and then copy out to sorted_*
    extern __shared__ float input_pvals_copy[];
    int threadblock_idx_start = blockIdx.x*blockDim.x;
    int max_idx = threadblock_idx_start + blockDim.x > input_pvals_length ? input_pvals_length-threadblock_idx_start-1 : blockDim.x;

    float tmp;
    int tmp_ordinal;

    int *input_pval_ordinals = (int *) (&input_pvals_copy[max_idx]);
	// if(threadIdx.x == 0) printf("input_pvals_copy: %ld, input_pval_ordinals: %ld\n", input_pvals_copy, input_pval_ordinals);
	// if(threadIdx.x == 0) printf("max_idx: %i, threadblock_idx_start: %i, blockDim.x: %i, input_pvals_length: %i\n", max_idx, threadblock_idx_start, blockDim.x, input_pvals_length);
    if(threadIdx.x <= max_idx){
      // The following reads should coalesce.
      input_pvals_copy[threadIdx.x] = input_pvals[threadIdx.x];
      input_pval_ordinals[threadIdx.x] = threadblock_idx_start+threadIdx.x; // ordinals are local for the moment
      __syncwarp();
      if(! threadIdx.x % CUDA_WARP_WIDTH){ // Co-sort value and input list order within the warp. 
                                         // Bubblesort actually works well here because we're only sorting up to 32 items
        for(int i = 0; i < (CUDA_WARP_WIDTH-1) && threadIdx.x/CUDA_WARP_WIDTH+i <= max_idx; ++i){
          int gi = threadIdx.x + i;
          for(int j = i+1; j < CUDA_WARP_WIDTH && threadIdx.x/CUDA_WARP_WIDTH+j <= max_idx; ++j){
            int gj = threadIdx.x + j;
            if(gj < input_pvals_length && input_pvals_copy[gi] > input_pvals_copy[gj]){
              tmp = input_pvals_copy[gi]; input_pvals_copy[gi] = input_pvals_copy[gj]; input_pvals_copy[gj] = tmp;
              tmp_ordinal = input_pval_ordinals[gi]; input_pval_ordinals[gi] = input_pval_ordinals[gj]; input_pval_ordinals[gj] = tmp_ordinal;
            }
          }
        }
      }
    }
    __syncthreads();
    // Merge co-sort up to CUDA_THREADBLOCK_MAX_THREADS items without recursion, all in one warp
    if(threadIdx.x < CUDA_WARP_WIDTH && threadIdx.x < max_idx){
      for(int half_sorting_swath = CUDA_WARP_WIDTH; half_sorting_swath <= input_pvals_length/2; half_sorting_swath*=2){
        int l1 = 0, l2 = 0, start_index = threadIdx.x*half_sorting_swath;
        if(start_index<input_pvals_length/2){
          while(l1 < half_sorting_swath && l2 < half_sorting_swath && half_sorting_swath + start_index + l2 < max_idx){
            if(input_pvals_copy[start_index + l1] <= input_pvals_copy[half_sorting_swath + start_index + l2]) {++l1;}
            else {
              tmp = input_pvals_copy[threadIdx.x + l1];
              input_pvals_copy[start_index + l1] = input_pvals_copy[half_sorting_swath + start_index + l2];
              input_pvals_copy[half_sorting_swath + start_index + l2] = tmp;
              tmp_ordinal = input_pval_ordinals[threadIdx.x + l1];
              input_pval_ordinals[start_index + l1] = input_pval_ordinals[half_sorting_swath + start_index + l2];
              input_pval_ordinals[half_sorting_swath + start_index + l2] = tmp_ordinal;
              l2++;
            }
          }
        }
        __syncwarp();
      }
    }

    if(threadIdx.x < max_idx){
      // Convert the local ordinals to global ordinals.
      sorted_pval_ordinals[threadIdx.x] = threadblock_idx_start+input_pval_ordinals[threadIdx.x];
      // Overwrite the original p-values stored in global memory with the sorted ones.
      input_pvals[threadblock_idx_start+threadIdx.x] = input_pvals_copy[threadIdx.x];

      // Assign tied values to the same rank (the minimum of their ranks otherwise). All subsequent ranks are elevated by the number of earlier ties. 
      int num_ranks = 1;
      if(! threadIdx.x){
        sorted_pval_ranks[0] = 1; // 1 because statistical ranks are 1-based, not 0-based like the arrays
        for(int i = 1; i < max_idx; i++){
          if(input_pvals_copy[i-1] != input_pvals_copy[i]){ // not a tie
            num_ranks++;
          }
		  if(input_pval_ordinals[i] >= input_pvals_length) printf("input_pval_ordinals at %i is greater than %i: %i\n", i, input_pvals_length, input_pval_ordinals[i]);
          sorted_pval_ranks[input_pval_ordinals[i]] = num_ranks;
        }
      }
    }
}
   
// False Discovery Rate (Benjamini-Hochberg multiple testing correction) given a set of independent p-values (a p-value is 
// the probability that the null hypothesis is true) calculated from applying some statistical test many times to different data.
// The effective_pvals_length argument allows us to correct as if the the total number of tests was effective_pvals_length.
__global__
void
fdr(float *input_pvals, int *input_pval_ranks, unsigned int input_pvals_length, float *output_qvals) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < input_pvals_length){
       // zero is special case of ranks below what the caller asked to report
       float qval = input_pval_ranks[idx] == 0 ? 1 : (double(input_pvals_length)/double(input_pval_ranks[idx]) * input_pvals[idx]);
       output_qvals[idx] = qval > 1 ? 1 : qval;
    }
}

// Sample the subject min distances (relative to some query as generated by find_colinear_anchor_candidates()) using a stride calculated to get exactly num_samples of them.
// Maximum allowed num_samples is CUDA_THREADBLOCK_MAX_THREADS, minimum is CUDA_WARP_WIDTH, and must be a power of two for the merge sort to work nicely.
__global__
void
get_min_distance_sorted_subject_sample(QTYPE_ACC *dtw_distances, int *min_idxs, int num_min_idxs, QTYPE_ACC *sorted_non_colinear_distances, int num_samples){
  extern __shared__ QTYPE_ACC sorting_list[];
  float tmp;

  // Grab the strided data in parallel
  if(threadIdx.x < num_samples){
    // Make the threadblock (CUDA_THREADBLOCK_MAX_THREADS) samples span the whole subject set of min indices to avoid as much subject and query locality bias as possible. 
    // This operation may take several 1000's cycles since it requires going to global memory twice, the memory access will not coalesce, and will 
    // almost definitely be L2 cache and TLB cache misses every time.
	sorting_list[threadIdx.x] = dtw_distances[min_idxs[threadIdx.x*num_min_idxs/num_samples]];

    // Sort
    __syncwarp();
    if(threadIdx.x % CUDA_WARP_WIDTH == 0){ // Sort CUDA_WARP_WIDTH item sublists with bubblesort (THE HORROR! ...but tons of latency in last double global memory fetch above since there is 
                            // no coalescence of memory access, so 
                            // every warp except the last one to complete has L1 read/write cycles to spare, for simplicity of single warp mergesort to follow).
      for(int i = 0; i < (CUDA_WARP_WIDTH-1) && (threadIdx.x + i) < num_samples; ++i){
        int gi = threadIdx.x + i;
        for(int j = i+1; j < CUDA_WARP_WIDTH && (threadIdx.x + j) < num_samples; ++j){
          int gj = threadIdx.x + j;
          if(sorting_list[gi] > sorting_list[gj]){
            tmp = sorting_list[gi]; sorting_list[gi] = sorting_list[gj]; sorting_list[gj] = tmp;
          }
        }
      }
    }

	__syncthreads();

    // Merge sort up to CUDA_THREADBLOCK_MAX_THREADS items without recursion, all in one warp
	bool end_hit = false;
	int new_start = 0;
    if(threadIdx.x < CUDA_WARP_WIDTH){
      for(int half_sorting_swath = CUDA_WARP_WIDTH; half_sorting_swath < num_samples; half_sorting_swath*=2){
        int l1 = 0, l2 = 0, start_index = threadIdx.x*half_sorting_swath;
        if(start_index<num_samples/2){
          while(l1 < half_sorting_swath && l2 < half_sorting_swath){
            if(sorting_list[start_index + l1] <= sorting_list[half_sorting_swath + start_index + l2]) {++l1;}
            else {
              tmp = sorting_list[threadIdx.x + l1]; 
              sorting_list[start_index + l1] = sorting_list[half_sorting_swath + start_index + l2]; 
              sorting_list[half_sorting_swath + start_index + l2] = tmp; 
              l2++;
			  if((half_sorting_swath + start_index + l2) == num_samples){
				l2 = 0;
				end_hit = true;
			  }
            }
          }
		  if(end_hit){
			new_start = start_index + l1;
		  }
        }
        __syncwarp();
      }
    }
    __syncthreads();

	if(threadIdx.x == 0){ // Sort CUDA_WARP_WIDTH item sublists with bubblesort (THE HORROR! ...but tons of latency in last double global memory fetch above since there is 
                            // no coalescence of memory access, so 
                            // every warp except the last one to complete has L1 read/write cycles to spare, for simplicity of single warp mergesort to follow).
      for(int i = new_start; i < num_samples; ++i){
        for(int j = i+1; j < num_samples; ++j){
          if(sorting_list[i] > sorting_list[j]){
            tmp = sorting_list[i]; sorting_list[i] = sorting_list[j]; sorting_list[j] = tmp;
          }
        }
      }
    }
	__syncthreads();
    // Parallel copy sorted result out to global memory
    sorted_non_colinear_distances[threadIdx.x] = sorting_list[threadIdx.x];
  }
}

// Host side function that loads the query into the GPU so either hard_dtw or soft_dtw can be run.
// query - the query that will be loaded to the GPU
// query_length - the length of te query
// normalization_mode - determines what type of normalization is performed on the query before loading to the GPU
// stream - CUDA kernel stream
__host__ __inline__
int
flash_dtw_setup(QTYPE* query, int query_length, int normalization_mode, cudaStream_t stream){
	// First, load the query into device constant memory as that's where the DTW kernel is expecting to find it later
    if(query_length > MAX_QUERY_SIZE){
      std::cerr << "Sorry, maximum query length as currently implemented in FLASH is " << MAX_QUERY_SIZE << ", requested " << query_length << std::endl;
      return 1;
    }
	// std::cerr << "Before normalization" << std::endl;
	// for(int i = 0; i < query_length; i++){
		// std::cerr << query[i] << " ";
	// }
	// std::cerr << std::endl;
    cudaMemcpyToSymbol(Gquery, query, sizeof(QTYPE)*query_length);                            CUERR("Copying query from CPU to GPU memory")
    //cudaMemcpyToSymbol(::Gquery_length, &query_length, sizeof(int), cudaMemcpyHostToDevice);                            CUERR("Copying query's length from CPU to GPU constant memory") 
    cudaMemcpyToSymbolAsync(::Gquery_length, &query_length, sizeof(int), 0, cudaMemcpyHostToDevice, stream);                             CUERR("Copying query's length from CPU to GPU constant memory") 
    float *mean, *stddev;
    QTYPE *min, *max;
    cudaMalloc(&mean, sizeof(float)); CUERR("Allocating GPU memory for single query mean"); 
    cudaMalloc(&stddev, sizeof(float)); CUERR("Allocating GPU memory for single query std dev");
    cudaMalloc(&min, sizeof(QTYPE)); CUERR("Allocating GPU memory for single query min");
    cudaMalloc(&max, sizeof(QTYPE)); CUERR("Allocating GPU memory for single query max");
    // 0 first arg = use the ::Gquery
    get_znorm_stats<QTYPE>((QTYPE*)0, query_length, mean, stddev, min, max);     CUERR("Setting single global query stats");
    dim3 norm_grid(DIV_ROUNDUP(query_length,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
	query_znorm<QTYPE><<<norm_grid, CUDA_THREADBLOCK_MAX_THREADS, 0, stream>>>(0, query_length, mean, stddev, min, max, normalization_mode);              CUERR("Applying global query normalization in flash_dtw_setup");

	// For debugging
    cudaStreamSynchronize(stream);              CUERR("Synchronizing stream after applying global query normalization");
    cudaMemcpyFromSymbol(query, Gquery, sizeof(QTYPE)*query_length);              CUERR("Copying global query from GPU to CPU");
	
	cudaFree(mean);			CUERR("Freeing GPU memory for single query mean in setup"); 
	cudaFree(stddev);			CUERR("Freeing GPU memory for single query stddev in setup"); 
	cudaFree(min);				CUERR("Freeing GPU memory for single query min in setup"); 
	cudaFree(max);				CUERR("Freeing GPU memory for single query max in setup"); 
	// std::cerr << "After normalization" << std::endl;
	// for(int i = 0; i < query_length; i++){
		// std::cerr << query[i] << " ";
	// }
	// std::cerr << std::endl;
    // std::cerr << "Normalized query (length" << query_length <<") : " << std::endl;
    // for(int j = 0; j < query_length; j++)
	// std::cerr << query[j] << " ";
    // std::cerr << std::endl;                          CUERR("Copying query's length from CPU to GPU constant memory") 
	return 0;
}

__host__
QTYPE*
Get_subArray(QTYPE* array, int array_length, int new_array_length){
	if(new_array_length > array_length){
		std::cerr << "New array length must be smaller than the original array. Exiting" << std::endl;
		return 0;
	}
	// QTYPE* new_array = (QTYPE*)malloc(new_array_length*sizeof(QTYPE));
	// memcpy(new_array, array, new_array_length);

	QTYPE* new_array = 0;
	cudaMalloc(&new_array, new_array_length*sizeof(QTYPE));		CUERR("Mallocing sub_array");
	cudaMemcpy(new_array, array, new_array_length*sizeof(QTYPE), cudaMemcpyDeviceToDevice);		CUERR("Copying to sub_array from array");

	return new_array;
}

/* Implement the lower bound search for bits of a long query, winnowing for co-linear candidate bits (below some random co-linearity probability), and return the co-linear anchors. */
// Host side function that performs the dtw calculation and pvalue calculation from a set of query data against a set of GPU loaded subject data
// query - the query to be compared against the GPU loaded subject
// query_length - the size of the query
// max_warp_proportion - warp proportion to be used for calculating the colinear margin
// max_pval - the maximum pvalue used to accept or reject a match. If a pvalue is found to be lower than this value, then it wont be counted
// max_qvalue - FDR limit for reporting matches
// max_ranks - the maximum number of ranks we want to calculate
// normalization_mode - determines what type of normalization is performed on the query before loading to the GPU
// results - buffer that will store all results from this function call. 
// record_anchors - flag to determine whether we want to store the anchors in the results buffer
// stream - CUDA kernel stream
__host__
int
flash_dtw(QTYPE *query, int query_length, char* query_name, float max_warp_proportion, float max_pvalue, float max_qvalue, int max_ranks, int normalization_mode, 
          match_record **results, int *num_results, int minidtw_size, int minidtw_warp, bool record_anchors, int use_fast_anchor_calc, int use_std, int use_hard_dtw, int verbose=0, cudaStream_t stream=0){

  if(query != 0){
	std::cerr << "Query was passed in so we will load it" << std::endl;
    if(flash_dtw_setup(query, query_length, normalization_mode, stream)){
		return 1;
	}
	// QTYPE* norm_query = (QTYPE*) malloc(sizeof(QTYPE)*query_length);
    // cudaMemcpyFromSymbol(norm_query, Gquery, sizeof(QTYPE)*query_length, 0, cudaMemcpyDeviceToHost); CUERR("Copying normalized query from GPU to CPU");
    
    // std::ofstream norm_file;
    // norm_file.open("gquery_output1.txt");
    // for(int i = 0; i < query_length; i++){
      // norm_file << norm_query[i] << std::endl;
    // }
    // norm_file.close();
	// free(norm_query);
  }
  else{
     int existing_query_length = 0;
     cudaMemcpyFromSymbol(&existing_query_length, ::Gquery_length, sizeof(int), 0, cudaMemcpyDeviceToHost);   CUERR("Checking for preloaded query in GPU memory, since the query in call to flash_dtw() was null");
	 if(existing_query_length == 0){
        std::cerr << "Sorry, no query was specified, and no existing query is in cache (e.g. a previous call to flash_dtw(), or load_and_segment_queries())" << std::endl;
        return 2;
     }
     query_length = existing_query_length;
  }
  // else reuse whatever query was loaded before

  // Check that there is a subject to search against
  if(!Dsubject){
    std::cerr << "Sorry, no subject has been loaded onto the GPU yet.  Please call set_subject() before flash_dtw()" << std::endl;
    return 3;
  }

  // Apply the mini DTWs all across the query against every reference position
  long existing_subject_length = 0;
  cudaMemcpyFromSymbol(&existing_subject_length, ::Tsubject_length, sizeof(QTYPE *));   CUERR("Checking preloaded subject length in GPU memory");
  
  long num_query_indices = query_length/minidtw_size;
  size_t query_mem_size = query_length*sizeof(QTYPE);
  size_t subject_mem_size = existing_subject_length*sizeof(QTYPE);
  
  long long *query_adjacent_candidates = NULL;
  size_t candidates_mem_size = sizeof(long long)*DIV_ROUNDUP(existing_subject_length, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices;
  cudaMalloc(&query_adjacent_candidates, candidates_mem_size);	CUERR("Allocating GPU memory for DTW anchor indices")
  QTYPE_ACC *query_adjacent_distances = NULL;
  size_t distances_mem_size = sizeof(QTYPE_ACC)*DIV_ROUNDUP(existing_subject_length, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices;
  cudaMalloc(&query_adjacent_distances, distances_mem_size);	CUERR("Allocating GPU memory for DTW anchor distances")
  std::cerr << "dist size: " << distances_mem_size << " so num of values is: " << distances_mem_size / sizeof(QTYPE_ACC) << std::endl;
  
  #if defined(_FLASH_UNIT_TEST)
  std::string output_path = "/export/home/sjhepbur/output/magenta_output/";
  output_path += query_name;
  output_path += "-";
  		
  std::string file_num = "2.txt";
  
  
  int norm_query_length = 0;
     cudaMemcpyFromSymbol(&norm_query_length, ::Gquery_length, sizeof(int), 0, cudaMemcpyDeviceToHost);   CUERR("Copy pre loaded query length");
  
  QTYPE* norm_query = (QTYPE*) malloc(norm_query_length*sizeof(QTYPE));
  cudaMemcpyFromSymbol(norm_query, Gquery, norm_query_length*sizeof(QTYPE));                        CUERR("Copying norm query buffer from GPU to CPU");
  
  // cudaMemcpyToSymbol(Gquery, query_values, sizeof(T)*query_length);                            CUERR("Copying Gquery from CPU to GPU emory")
  
  std::ofstream norm_file;
  norm_file.open(output_path + (std::string("normalized_query_file_test") + file_num).c_str());
  			
  for(int i = 0; i < norm_query_length; i++){
  	norm_file << i << " - " << norm_query[i] << std::endl;
  }
  			
  norm_file.close();
  #endif
  
  

  // TODO got here, no more memory limit, do it in one kernel, calculate griddim properly (accounting for extended subjects)
  int threadblock_size_dtw = CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE; 
  dim3 griddim_dtw(DIV_ROUNDUP(existing_subject_length, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices, 1, 1);
  
  if(minidtw_size < 10 || minidtw_size > 20) {
	std::cerr << "Error: minidtw_size is out of bounds, must be between 10 and 20" << std::endl;
	exit(1);
  }
  if(minidtw_warp < 2 || minidtw_warp > 4) {
	std::cerr << "Error: minidtw_warp is out of bounds, mist be between 2 and 4" << std::endl;
	exit(1);
  }
  if(verbose) printf("\nMemory: \nQuery: %lukB\nSubject: %luMB\nCandidates: %luMB\nDistances: %luMB\n", (unsigned long)query_mem_size/1024, (unsigned long)subject_mem_size/1024/1024, (unsigned long)candidates_mem_size/1024/1024, (unsigned long)distances_mem_size/1024/1024);
  if(verbose) printf("indices: %li, padded indices: %li\n", existing_subject_length*num_query_indices, DIV_ROUNDUP(existing_subject_length, 1024/MINIDTW_STRIDE)*1024/MINIDTW_STRIDE*num_query_indices); 
  if(verbose) printf("running kernel %sdtw%s<%i,%i><<<(%i, %i, %i), %i>>>\n", (minidtw_size == 10 && use_hard_dtw == 1) ? "hard_" : "soft_", use_std ? "_std" : "", minidtw_size, minidtw_warp, griddim_dtw.x, griddim_dtw.y, griddim_dtw.z, threadblock_size_dtw); 
  if(minidtw_size == 10 && minidtw_warp == 2 && use_hard_dtw == 1) {
	if(use_std) {
	  hard_dtw<<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances, minidtw_size); CUERR("Running DTW anchor distance calculations")
    } else {
	  hard_dtw<<<griddim_dtw, threadblock_size_dtw, 0, stream>>>(num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances, minidtw_size); CUERR("Running DTW anchor distance calculations")
	}
  } else {
	soft_dtw_wrap(griddim_dtw, threadblock_size_dtw, stream, num_query_indices, Dsubject, Dsubject_std, query_adjacent_candidates, query_adjacent_distances, minidtw_size, minidtw_warp, use_std);
  }
  cudaStreamSynchronize(stream);
  
  // QTYPE_ACC* return_dist_buff = (QTYPE_ACC*) malloc(distances_mem_size);
  // cudaMemcpy(return_dist_buff, query_adjacent_distances, distances_mem_size, cudaMemcpyDeviceToHost);                        CUERR("Copying dist buffer from GPU to CPU");
  // std::cerr << "Distances(" << griddim_dtw.x << "): " << std::endl;
  // for(int i = 0; i < griddim_dtw.x; i++){
     // std::cerr  << i << " - " << std::fixed << return_dist_buff[i] << std::endl;
  // }
  // free(return_dist_buff);
  
  #if defined(_FLASH_UNIT_TEST)
  // std::string output_path = "/export/home/sjhepbur/output/magenta_output/";
  // output_path += query_name;
  // output_path += "-";
  
  long long* return_query_buff = (long long*) malloc(candidates_mem_size);
  cudaMemcpy(return_query_buff, query_adjacent_candidates, candidates_mem_size, cudaMemcpyDeviceToHost);                        CUERR("Copying query buffer from GPU to CPU");
  
  // int subsequence_length = 0;
  // long long* query_adjacent_candidates_subsequence = longestIncreasingSubsequence<long long>(return_query_buff, griddim_dtw.x, &subsequence_length);
  
  QTYPE_ACC* return_dist_buff = (QTYPE_ACC*) malloc(distances_mem_size);
  cudaMemcpy(return_dist_buff, query_adjacent_distances, distances_mem_size, cudaMemcpyDeviceToHost);                        CUERR("Copying dist buffer from GPU to CPU");
  
  // std::string file_num = "2.txt";
  
  // std::ofstream subsequence_file;
  // subsequence_file.open((std::string("subsequence_file_test") + file_num).c_str());
  
  std::ofstream query_file;
  query_file.open(output_path + (std::string("query_file_test") + file_num).c_str());
  
  std::ofstream dist_file;
  dist_file.open(output_path + (std::string("dist_file_test") + file_num).c_str());
  
  for(int i = 0; i < griddim_dtw.x; i++){
    query_file << i << " - " << return_query_buff[i] << std::endl;
    dist_file << i << " - " << std::fixed << return_dist_buff[i] << std::endl;
  }
  // for(int i = 0; i < subsequence_length; i++){
    // subsequence_file << query_adjacent_candidates_subsequence[i] << std::endl;
  // }
  // subsequence_file.close();
  query_file.close();
  dist_file.close();
  free(return_query_buff);
  free(return_dist_buff);
  #endif
  
  // Set up to find co-linear results of the mini DTWs within each dtw_results section (defined by nTPB above) that are loaded into a threadblock
  int threadblock_size = int(num_query_indices);
  if(threadblock_size > CUDA_THREADBLOCK_MAX_THREADS){
    std::cerr << "Error: query of " << num_query_indices << " is too long for calc_anchor_candidates_colinear_pvals kernel\n";
	exit(1);
  }
  
  // TODO Check memory
  size_t num_subject_buckets_per_query_minidtw = DIV_ROUNDUP(existing_subject_length/MINIDTW_STRIDE, CUDA_THREADBLOCK_MAX_THREADS);
  // num_subject_buckets_per_query_minidtw = 10;

  QTYPE_ACC *global_non_colinear_distances;
  int *num_non_colinear_distances;
  int expected_num_members;
  if(use_fast_anchor_calc) {
	cudaMalloc(&global_non_colinear_distances, sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS);		CUERR("Allocating GPU memory for global_non_colinear_distances fast anchor");
	cudaMalloc(&num_non_colinear_distances, sizeof(int));																						CUERR("Allocating GPU memory for num_non_colinear_distances");
	load_non_colinear_distances<<<1, num_query_indices, 0, stream>>>(query_adjacent_distances, num_query_indices, num_subject_buckets_per_query_minidtw, global_non_colinear_distances, num_non_colinear_distances);
  } else {
	// this value controls how much memory is used. if this allocation fails, it needs to be reduced (ie. set it to 10) which will bring the memory under control but take longer for calc_anchor_candidates to finish
	expected_num_members = num_query_indices; // TODO automatically configure this to not run out of memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem,&total_mem); 		CUERR("Checking remaining memory on GPU");
	if((sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*num_query_indices) > free_mem){
		num_subject_buckets_per_query_minidtw = free_mem;
		int max = num_subject_buckets_per_query_minidtw;
		int start = 0;
		while(num_subject_buckets_per_query_minidtw != start){
			//std::cerr << "Mem trying to be allocated: " << sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*expected_num_members << " and num_subject_buckets_per_query_minidtw: " << num_subject_buckets_per_query_minidtw << std::endl;
			if(cudaMalloc(&global_non_colinear_distances, sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*expected_num_members) != 0){
				cudaGetLastError();
				max = num_subject_buckets_per_query_minidtw;
				num_subject_buckets_per_query_minidtw = (num_subject_buckets_per_query_minidtw + start) / 2; 
			} else{
				cudaFree(global_non_colinear_distances); CUERR ("Freeing global_non_colinear_distances during setup");
				if(max == num_subject_buckets_per_query_minidtw)
					break;
				start = num_subject_buckets_per_query_minidtw;
				num_subject_buckets_per_query_minidtw += (max - num_subject_buckets_per_query_minidtw) / 2;
			}
		}
		size_t tot_mem_to_malloc = sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*expected_num_members;
		// Multiplying reserve memory by 1.5 to account for malloc only mallocing only on memory word boundaries, therefore more space is needed for these allocations
		tot_mem_to_malloc = tot_mem_to_malloc - 1.5*(2*sizeof(unsigned int) + sizeof(float)*MAX_PVAL_MATCHES_KEPT + 2*sizeof(int)*MAX_PVAL_MATCHES_KEPT + 2*sizeof(long)*MAX_PVAL_MATCHES_KEPT);
		if(record_anchors) tot_mem_to_malloc = tot_mem_to_malloc - 1.5*sizeof(int)*MAX_PVAL_MATCHES_KEPT;
		std::cerr << "num_subject_buckets_per_query_minidtw before correction: " << num_subject_buckets_per_query_minidtw << " and tot_mem_to_malloc: " << tot_mem_to_malloc << std::endl;
		#if defined(_FLASH_UNIT_TEST)
		tot_mem_to_malloc = tot_mem_to_malloc - 1.5*(5*sizeof(int)*num_subject_buckets_per_query_minidtw + sizeof(long long)*num_subject_buckets_per_query_minidtw + 2*sizeof(QTYPE_ACC)*num_subject_buckets_per_query_minidtw + sizeof(float)*num_subject_buckets_per_query_minidtw + 2*sizeof(long)*num_subject_buckets_per_query_minidtw);
		#endif
		num_subject_buckets_per_query_minidtw = tot_mem_to_malloc/(sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*expected_num_members);
		std::cerr << "num_subject_buckets_per_query_minidtw after correction: " << num_subject_buckets_per_query_minidtw << " sizeof(QTYPE_ACC): " << sizeof(QTYPE_ACC) << " CUDA_THREADBLOCK_MAX_THREADS: " << CUDA_THREADBLOCK_MAX_THREADS << " expected_num_members: " << expected_num_members << std::endl;
		cudaMalloc(&global_non_colinear_distances, sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*expected_num_members); CUERR("Allocating GPU memory for global_non_colinear_distances after reducing bucket size");
	} else{
		cudaMalloc(&global_non_colinear_distances, sizeof(QTYPE_ACC)*CUDA_THREADBLOCK_MAX_THREADS*num_subject_buckets_per_query_minidtw*expected_num_members);		CUERR("Allocating GPU memory for global_non_colinear_distances");
	}
  }
  
  dim3 pval_griddim(num_subject_buckets_per_query_minidtw, 1, 1);
  size_t required_threadblock_shared_memory = num_query_indices*(2*sizeof(QTYPE_ACC)+sizeof(long long)+sizeof(short)+3*sizeof(int))+sizeof(int)+sizeof(short);
  
  // Allocate space for the co-linearity p-values for each region against the query
  float *anchor_pvals;
  int *anchors_colinear_num_members = 0;
  int *anchors_leftmost_query, *anchors_rightmost_query;
  long *anchors_leftmost_subject, *anchors_rightmost_subject;
  unsigned int *num_kept_anchor_pvals = 0;
  unsigned int *num_discarded_anchor_pvals = 0;
  cudaMalloc(&num_kept_anchor_pvals, sizeof(unsigned int));                                   CUERR("Allocating GPU memory for DTW anchor kept count");
  cudaMalloc(&num_discarded_anchor_pvals, sizeof(unsigned int));                              CUERR("Allocating GPU memory for DTW anchor discarded count");
  cudaMalloc(&anchor_pvals, sizeof(float)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for DTW anchor p-values");
  if(record_anchors)cudaMalloc(&anchors_colinear_num_members, sizeof(int)*MAX_PVAL_MATCHES_KEPT);              CUERR("Allocating GPU memory for set membership");
  cudaMalloc(&anchors_leftmost_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' first DTW anchors query");
  cudaMalloc(&anchors_rightmost_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' last DTW anchors query");
  cudaMalloc(&anchors_leftmost_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' first DTW anchors subject");
  cudaMalloc(&anchors_rightmost_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' last DTW anchors subject");

 
  //cudaFuncSetCacheConfig(calc_anchor_candidates_colinear_pvals, cudaFuncCachePreferL1);       CUERR("Setting L1 cache preference for p-value kernel call");
  // This is the real workhorse kernel call

  int* all_memberships = 0;
  int* lis_memberships = 0;
  long long* all_subject_indices = 0;

  QTYPE_ACC* colinear_buff = 0;
  QTYPE_ACC* non_colinear_buff = 0;

  int* anch_mem_buff = 0;
  float* pval_buff = 0;
  int* left_query_buff = 0;
  int* right_query_buff = 0;
  long* left_subject_buff = 0;
  long* right_subject_buff = 0;

  // Don't really need these buffers if we aren't testing
  #if defined(_FLASH_UNIT_TEST)

  int buffer_size = num_query_indices*pval_griddim.x;

  cudaMalloc(&all_memberships, sizeof(int)*buffer_size);				CUERR("Allocating GPU memory for membership buffer");
  cudaMalloc(&lis_memberships, sizeof(int)*buffer_size);				CUERR("Allocating GPU memory for lis membership buffer");
  cudaMalloc(&all_subject_indices, sizeof(long long)*buffer_size);	CUERR("Allocating GPU memory for subject indices buffer");
  
  cudaMalloc(&colinear_buff, sizeof(QTYPE_ACC)*buffer_size);			CUERR("Allocating GPU memory for colinear buffer");
  cudaMalloc(&non_colinear_buff, sizeof(QTYPE_ACC)*buffer_size);		CUERR("Allocating GPU memory for colinear buffer");
  
  cudaMalloc(&anch_mem_buff, sizeof(int)*buffer_size);				CUERR("Allocating GPU memory for anchor member buffer");
  cudaMalloc(&pval_buff, sizeof(float)*buffer_size);					CUERR("Allocating GPU memory for pvalue buffer");
  cudaMalloc(&left_query_buff, sizeof(int)*buffer_size);				CUERR("Allocating GPU memory for left query buffer");
  cudaMalloc(&right_query_buff, sizeof(int)*buffer_size);				CUERR("Allocating GPU memory for right query buffer");
  cudaMalloc(&left_subject_buff, sizeof(long)*buffer_size);			CUERR("Allocating GPU memory for left subject buffer");
  cudaMalloc(&right_subject_buff, sizeof(long)*buffer_size);			CUERR("Allocating GPU memory for right subject buffer");
  #endif

  if(verbose) printf("running kernel %scalc_anchor_candidates_colinear_pvals<<<(%i, %i, %i), %i, %zi>>>\n", use_fast_anchor_calc ? "fast_" : "through_", pval_griddim.x, pval_griddim.y, pval_griddim.z, threadblock_size, required_threadblock_shared_memory);
  if(use_fast_anchor_calc) {
	fast_calc_anchor_candidates_colinear_pvals<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, stream>>>(
								query_adjacent_candidates, num_query_indices, num_subject_buckets_per_query_minidtw,
								global_non_colinear_distances, num_non_colinear_distances, 
								max_warp_proportion, max_pvalue, query_adjacent_distances, 
								num_kept_anchor_pvals, num_discarded_anchor_pvals, MAX_PVAL_MATCHES_KEPT, 
								anchor_pvals, anchors_leftmost_query, anchors_rightmost_query, anchors_leftmost_subject, anchors_rightmost_subject, anchors_colinear_num_members,
								minidtw_size, all_memberships, all_subject_indices, lis_memberships, colinear_buff, non_colinear_buff, anch_mem_buff, pval_buff, left_query_buff, right_query_buff, left_subject_buff, right_subject_buff);   CUERR("Running p-value calculations");
  } else {
	thorough_calc_anchor_candidates_colinear_pvals<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, stream>>>(
								query_adjacent_candidates, num_query_indices, num_subject_buckets_per_query_minidtw,
								global_non_colinear_distances, expected_num_members, 
								max_warp_proportion, max_pvalue, query_adjacent_distances, 
								num_kept_anchor_pvals, num_discarded_anchor_pvals, MAX_PVAL_MATCHES_KEPT, 
								anchor_pvals, anchors_leftmost_query, anchors_rightmost_query, anchors_leftmost_subject, anchors_rightmost_subject, anchors_colinear_num_members,
								minidtw_size, all_memberships, all_subject_indices, lis_memberships, colinear_buff, non_colinear_buff, anch_mem_buff, pval_buff, left_query_buff, right_query_buff, left_subject_buff, right_subject_buff);   CUERR("Running p-value calculations");
  }


  #if defined(_FLASH_UNIT_TEST)

  int* host_all_memberships = (int*)malloc(sizeof(int)*buffer_size);
  int* host_lis_memberships = (int*)malloc(sizeof(int)*buffer_size);
  long long* host_all_subject_indices = (long long*)malloc(sizeof(long long)*buffer_size);
  
  QTYPE_ACC* return_colinear_buff = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*buffer_size);
  QTYPE_ACC* return_non_colinear_buff = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*buffer_size);
  
  int* return_anch_mem_buff = (int*) malloc(sizeof(int)*buffer_size);
  float* return_pval_buff = (float*) malloc(sizeof(float)*buffer_size);
  int* return_left_query_buff = (int*) malloc(sizeof(int)*buffer_size);
  int* return_right_query_buff = (int*) malloc(sizeof(int)*buffer_size);
  long* return_left_subject_buff = (long*) malloc(sizeof(long)*buffer_size);
  long* return_right_subject_buff = (long*) malloc(sizeof(long)*buffer_size);
  
  cudaMemcpy(host_all_memberships, all_memberships, sizeof(int)*buffer_size, cudaMemcpyDeviceToHost);						CUERR("Copying membership buffer from GPU to CPU");
  cudaMemcpy(host_lis_memberships, lis_memberships, sizeof(int)*buffer_size, cudaMemcpyDeviceToHost);						CUERR("Copying lis memebrship buffer from GPU to CPU");
  cudaMemcpy(host_all_subject_indices, all_subject_indices, sizeof(long long)*buffer_size, cudaMemcpyDeviceToHost);		CUERR("Copying subject indices buffer from GPU to CPU");
  
  cudaMemcpy(return_colinear_buff, colinear_buff, sizeof(QTYPE_ACC)*buffer_size, cudaMemcpyDeviceToHost);					CUERR("Copying colinear buffer from GPU to CPU");
  cudaMemcpy(return_non_colinear_buff, non_colinear_buff, sizeof(QTYPE_ACC)*buffer_size, cudaMemcpyDeviceToHost);			CUERR("Copying non colinear buffer from GPU to CPU");
  
  cudaMemcpy(return_anch_mem_buff, anch_mem_buff, sizeof(int)*buffer_size, cudaMemcpyDeviceToHost);						CUERR("Copying anchor member buffer from GPU to CPU");
  cudaMemcpy(return_pval_buff, pval_buff, sizeof(float)*buffer_size, cudaMemcpyDeviceToHost);								CUERR("Copying pvalue buffer from GPU to CPU");
  cudaMemcpy(return_left_query_buff, left_query_buff, sizeof(int)*buffer_size, cudaMemcpyDeviceToHost);					CUERR("Copying left query buffer from GPU to CPU");
  cudaMemcpy(return_right_query_buff, right_query_buff, sizeof(int)*buffer_size, cudaMemcpyDeviceToHost);					CUERR("Copying right query buffer from GPU to CPU");
  cudaMemcpy(return_left_subject_buff, left_subject_buff, sizeof(long)*buffer_size, cudaMemcpyDeviceToHost);				CUERR("Copying left subject buffer from GPU to CPU");
  cudaMemcpy(return_right_subject_buff, right_subject_buff, sizeof(long)*buffer_size, cudaMemcpyDeviceToHost);			CUERR("Copying right subject buffer from GPU to CPU");
  
  std::ofstream mem_file;
  std::ofstream lis_file;
  std::ofstream indices_file;
  
  std::ofstream colinear_file;
  std::ofstream non_colinear_file;
  
  std::ofstream anch_mem_file;
  std::ofstream pvalue_file;
  std::ofstream left_query_file;
  std::ofstream right_query_file;
  std::ofstream left_subject_file;
  std::ofstream right_subject_file;
  
  mem_file.open(output_path + (std::string("mem_file_test") + file_num).c_str());
  lis_file.open(output_path + (std::string("lis_file_test") + file_num).c_str());
  indices_file.open(output_path + (std::string("indices_file_test") + file_num).c_str());
  
  colinear_file.open(output_path + (std::string("colinear_file_test") + file_num).c_str());
  non_colinear_file.open(output_path + (std::string("non_colinear_file_test") + file_num).c_str());
  
  anch_mem_file.open(output_path + (std::string("anch_mem_file") + file_num).c_str());
  pvalue_file.open(output_path + (std::string("pvalue_test") + file_num).c_str());
  left_query_file.open(output_path + (std::string("left_query_test") + file_num).c_str());
  right_query_file.open(output_path + (std::string("right_query_test") + file_num).c_str());
  left_subject_file.open(output_path + (std::string("left_subject_test") + file_num).c_str());
  right_subject_file.open(output_path + (std::string("right_subject_test") + file_num).c_str());
  
  // We write results to a file if we want to test the output
  for(int i = 0; i < buffer_size; i++){
lis_file << i << " - " << host_lis_memberships[i] << std::endl;
mem_file << i << " - " << host_all_memberships[i] << std::endl;
indices_file << i << " - " << host_all_subject_indices[i] << std::endl;
  
colinear_file << i << " - " << std::fixed << return_colinear_buff[i] << std::endl;
non_colinear_file << i << " - " << std::fixed << return_non_colinear_buff[i] << std::endl;
  
anch_mem_file << i << " - " << return_anch_mem_buff[i] << std::endl;
pvalue_file << i << " - " << return_pval_buff[i] << std::endl;
left_query_file << i << " - " << return_left_query_buff[i] << std::endl;
right_query_file << i << " - " << return_right_query_buff[i] << std::endl;
left_subject_file << i << " - " << return_left_subject_buff[i] << std::endl;
right_subject_file << i << " - " << return_right_subject_buff[i] << std::endl;
  }
  
  mem_file.close();
  lis_file.close();
  indices_file.close();
  
  colinear_file.close();
  non_colinear_file.close();
  
  anch_mem_file.close();
  pvalue_file.close();
  left_query_file.close();
  right_query_file.close();
  left_subject_file.close();
  right_subject_file.close();
  
  cudaFree(all_memberships);			CUERR("Freeing GPU memory for membership buffer");
  cudaFree(lis_memberships);			CUERR("Freeing GPU memory for lis membership buffer");
  cudaFree(all_subject_indices);		CUERR("Freeing GPU memory for subject indices buffer");
  
  cudaFree(colinear_buff);			CUERR("Freeing GPU memory for colinear buffer");
  cudaFree(non_colinear_buff);		CUERR("Freeing GPU memory for non colinear buffer");
  
  cudaFree(anch_mem_buff);			CUERR("Freeing GPU memory for anchor member buffer");
  cudaFree(pval_buff);				CUERR("Freeing GPU memory for pvalue buffer");
  cudaFree(left_query_buff);			CUERR("Freeing GPU memory for left query buffer");
  cudaFree(right_query_buff);			CUERR("Freeing GPU memory for right query buffer");
  cudaFree(left_subject_buff);		CUERR("Freeing GPU memory for left subject buffer");
  cudaFree(right_subject_buff);		CUERR("Freeing GPU memory for right subject buffer");
  
  free(host_all_memberships);
  free(host_lis_memberships);
  free(host_all_subject_indices);
  
  free(return_colinear_buff);
  free(return_non_colinear_buff);
  
  free(return_anch_mem_buff);
  free(return_pval_buff);
  free(return_left_query_buff);
  free(return_right_query_buff);
  free(return_left_subject_buff);
  free(return_right_subject_buff);
	
  #endif
  

  // By far the biggest memory we'll allocate in this method, free it ASAP.
  if(use_fast_anchor_calc)
	cudaFree(num_non_colinear_distances);																CUERR("Freeing GPU memory for num_non_colinear_distances");
  cudaFree(query_adjacent_candidates);																	CUERR("Freeing GPU memory for query_adjacent_candidates");
  cudaFree(query_adjacent_distances);																	CUERR("Freeing GPU memory for query_adjacent_distances");
  cudaFree(global_non_colinear_distances);																CUERR("Freeing GPU memory for global_non_colinear_distances");

  // TODO: join subject-and-query adjacent colinear sets into a joint set as they are only separate because of the memory limits of the calc_anchor_candidates_colinear_pvals() kernel
  //merge_colinear_pvals<<<1,stream>>>(anchor_pvals, num_kept_anchor_pvals, anchors_leftmost, anchors_rightmost, anchors_colinear_num_members);

  unsigned int num_pvals_host;

  cudaMemcpyAsync(&num_pvals_host, num_kept_anchor_pvals, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream); CUERR("Copying number of p-values passing criterion from GPU to CPU");
  cudaStreamSynchronize(stream);
  float *anchor_pvals_host = 0;
  float *placeholder_anchor_pvals_host = 0;
  int *anchors_colinear_num_members_host = 0;
  int *anchors_leftmost_query_host = 0;
  int *anchors_rightmost_query_host = 0;
  long *anchors_leftmost_subject_host = 0;
  long *anchors_rightmost_subject_host = 0;
  
  int *placeholder_anchors_leftmost_query_host = 0;
  int *placeholder_anchors_rightmost_query_host = 0;
  long *placeholder_anchors_leftmost_subject_host = 0;

  // Did the caller want to limit the reported matches by FDR or rank?
  float *qvals = 0;
  int *anchor_pval_ranks = 0;
  int *anchor_pval_ranks_host = 0;
  int *anchor_pval_ordinals = 0;
  if(verbose) printf("num_pvals_host: %i\n", num_pvals_host);
  if(num_pvals_host == 0){
    *num_results = 0;
  }
  else{

    // Initiate the asynchronous copy to host of the match information we have so we can hide the PCI bus latency if a rank criterion 
    // or FDR was specified and further calculation is necessary (but doesn't affect these values being copied).
    // Note that if FDR or # ranks criteria are strict, we may be copying a bunch of data back that won't get passed back to the caller,
    // but this is our design choice: to value potential latency reduction over minimizing pinned host memory usage.
    cudaMallocHost(&anchor_pvals_host, num_pvals_host*sizeof(float));                             CUERR("Allocating CPU memory for DTW anchor p-values");
    cudaMallocHost(&placeholder_anchor_pvals_host, num_pvals_host*sizeof(float));                             CUERR("Allocating CPU memory for DTW anchor p-value placeholder");
	cudaMallocHost(&anchors_leftmost_query_host, sizeof(int)*num_pvals_host);                             CUERR("Allocating CPU memory for sets' first query DTW anchors");
    cudaMallocHost(&anchors_leftmost_subject_host, sizeof(long)*num_pvals_host);                             CUERR("Allocating CPU memory for sets' first subject DTW anchors");
    cudaMallocHost(&anchors_rightmost_query_host, sizeof(int)*num_pvals_host);                             CUERR("Allocating CPU memory for sets' last query DTW anchors");
    cudaMallocHost(&anchors_rightmost_subject_host, sizeof(long)*num_pvals_host);                             CUERR("Allocating CPU memory for sets' last subject DTW anchors");
	
	cudaMallocHost(&placeholder_anchors_leftmost_query_host, sizeof(int)*num_pvals_host);                        CUERR("Allocating CPU memory for sets' first query DTW anchors placeholder");
	cudaMallocHost(&placeholder_anchors_rightmost_query_host, sizeof(int)*num_pvals_host);                       CUERR("Allocating CPU memory for sets' last query DTW anchors placeholder");
	cudaMallocHost(&placeholder_anchors_leftmost_subject_host, sizeof(long)*num_pvals_host);                       CUERR("Allocating CPU memory for sets' last query DTW anchors placeholder");
	
    cudaMemcpyAsync(anchor_pvals_host, anchor_pvals, sizeof(float)*num_pvals_host, cudaMemcpyDeviceToHost, stream);  CUERR("Copying DTW anchor p-values from GPU to CPU");
    cudaMemcpyAsync(anchors_leftmost_query_host, anchors_leftmost_query, sizeof(int)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying sets' first DTW query anchors from GPU to CPU");
    cudaMemcpyAsync(anchors_rightmost_query_host, anchors_rightmost_query, sizeof(int)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying sets' last DTW query anchors from GPU to CPU");
    cudaMemcpyAsync(anchors_leftmost_subject_host, anchors_leftmost_subject, sizeof(long)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying sets' first DTW subject anchors from GPU to CPU");
    cudaMemcpyAsync(anchors_rightmost_subject_host, anchors_rightmost_subject, sizeof(long)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying sets' last DTW subject anchors from GPU to CPU");
    if(record_anchors){
      cudaMallocHost(&anchors_colinear_num_members_host, sizeof(int)*num_pvals_host);              CUERR("Allocating GPU memory for set membership");
      cudaMemcpyAsync(anchors_colinear_num_members_host, anchors_colinear_num_members, sizeof(int)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying set membership from GPU to CPU");
    }
	
	
	#if defined(_FLASH_UNIT_TEST)
	std::ofstream anch_pval_host_file;
	std::ofstream anch_left_que_host_file;
	std::ofstream anch_right_que_host_file;
	std::ofstream anch_left_sub_host_file;
	std::ofstream anch_right_sub_host_file;
    
	anch_pval_host_file.open(output_path + (std::string("anch_pval_nonsort_test") + file_num).c_str());
	anch_left_que_host_file.open(output_path + (std::string("anch_left_que_nonsort_test") + file_num).c_str());
	anch_right_que_host_file.open(output_path + (std::string("anch_right_que_nonsort_test") + file_num).c_str());
	anch_left_sub_host_file.open(output_path + (std::string("anch_left_sub_nonsort_test") + file_num).c_str());
	anch_right_sub_host_file.open(output_path + (std::string("anch_right_sub_nonsort_test") + file_num).c_str());
	
	// We write results to a file if we want to test the output
	for(int i = 0; i < num_pvals_host; i++){
		anch_pval_host_file << i << " - " << anchor_pvals_host[i] << std::endl;
		anch_left_que_host_file << i << " - " << anchors_leftmost_query_host[i] << std::endl;
		anch_right_que_host_file << i << " - " << anchors_rightmost_query_host[i] << std::endl;
		anch_left_sub_host_file << i << " - " << anchors_leftmost_subject_host[i] << std::endl;
		anch_right_sub_host_file << i << " - " << anchors_rightmost_subject_host[i] << std::endl;
	}
	
	anch_pval_host_file.close();
	anch_left_que_host_file.close();
	anch_right_que_host_file.close();
	anch_left_sub_host_file.close();
	anch_right_sub_host_file.close();
	#endif

    if(max_qvalue >= 0 || max_ranks > 0){
      dim3 fdr_griddim(DIV_ROUNDUP(num_pvals_host, CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
      // Rank the matches, then calculate the false discovery rate for the matches if required (pval -> qval)
      threadblock_size = num_pvals_host > CUDA_THREADBLOCK_MAX_THREADS ? CUDA_THREADBLOCK_MAX_THREADS : num_pvals_host;
      
	  int anchor_num_ranks;
	  cudaMalloc(&anchor_pval_ordinals, sizeof(int)*num_pvals_host);                                         CUERR("Allocating GPU memory for DTW anchor p-value ordinals");
	  cudaMalloc(&anchor_pval_ranks, sizeof(int)*num_pvals_host);											 CUERR("Allocating GPU memory for DTW anchor p-value ranks");
	  cudaMallocHost(&anchor_pval_ranks_host, sizeof(int)*num_pvals_host);											 CUERR("Allocating CPU memory for DTW anchor p-value ranks");
	  
	  cudaStreamSynchronize(stream);
	  cudaMemcpy(placeholder_anchor_pvals_host, anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToHost);
	  if(record_anchors){
		thrust::sort_by_key(anchor_pvals_host, anchor_pvals_host+num_pvals_host, anchors_colinear_num_members_host);
	    cudaMemcpy(anchor_pvals_host, placeholder_anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToHost);
	  }
	  thrust::sort_by_key(anchor_pvals_host, anchor_pvals_host+num_pvals_host, anchors_leftmost_query_host);
	  cudaMemcpy(anchor_pvals_host, placeholder_anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToHost);
	  thrust::sort_by_key(anchor_pvals_host, anchor_pvals_host+num_pvals_host, anchors_leftmost_subject_host);
	  cudaMemcpy(anchor_pvals_host, placeholder_anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToHost);
	  thrust::sort_by_key(anchor_pvals_host, anchor_pvals_host+num_pvals_host, anchors_rightmost_query_host);
	  cudaMemcpy(anchor_pvals_host, placeholder_anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToHost);
	  thrust::sort_by_key(anchor_pvals_host, anchor_pvals_host+num_pvals_host, anchors_rightmost_subject_host);
	  cudaStreamSynchronize(stream);
	  cudaMemcpyAsync(anchor_pvals, anchor_pvals_host, num_pvals_host*sizeof(float), cudaMemcpyHostToDevice, stream);
	  
	  int start = 0;
	  int end = 0;
	  float same_pval = anchor_pvals_host[0];
	  for(int i = 1; i < num_pvals_host; i++){
		int total_size = end - start;
		 if((anchor_pvals_host[i] != same_pval || i == num_pvals_host-1) && total_size > 0){
		  if(i == num_pvals_host-1) end = i;
		  cudaMemcpy(placeholder_anchors_leftmost_query_host, anchors_leftmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
		  if(record_anchors){
		    thrust::sort_by_key(anchors_leftmost_query_host+start, anchors_leftmost_query_host+end+1, anchors_colinear_num_members_host+start);
	        cudaMemcpy(anchors_leftmost_query_host, placeholder_anchors_leftmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
	      }
		  thrust::sort_by_key(anchors_leftmost_query_host+start, anchors_leftmost_query_host+end+1, anchors_leftmost_subject_host+start);
		  cudaMemcpy(anchors_leftmost_query_host, placeholder_anchors_leftmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
		  thrust::sort_by_key(anchors_leftmost_query_host+start, anchors_leftmost_query_host+end+1, anchors_rightmost_query_host+start);
		  cudaMemcpy(anchors_leftmost_query_host, placeholder_anchors_leftmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
		  thrust::sort_by_key(anchors_leftmost_query_host+start, anchors_leftmost_query_host+end+1, anchors_rightmost_subject_host+start);
		  cudaStreamSynchronize(stream);		  
			  
		  same_pval = anchor_pvals_host[i];
		  start = i;
		  end = i;
		} else if(anchor_pvals_host[i] == same_pval){
		  end = i;
		} else{
		  same_pval = anchor_pvals_host[i];
		  start = i;
		  end = i;
		}
	  }
	
	  start = 0;
	  end = 0;
	  int same_anchor_left_query = anchors_leftmost_query_host[0];
	  for(int i = 1; i < num_pvals_host; i++){
		int total_size = end - start;
		if((anchors_leftmost_query_host[i] != same_anchor_left_query || i == num_pvals_host-1) && total_size > 0){
		  if(i == num_pvals_host-1) end = i;
		  cudaMemcpy(placeholder_anchors_rightmost_query_host, anchors_rightmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
		  if(record_anchors){
		    thrust::sort_by_key(anchors_rightmost_query_host+start, anchors_rightmost_query_host+end+1, anchors_colinear_num_members_host+start);
	        cudaMemcpy(anchors_rightmost_query_host, placeholder_anchors_rightmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
	      }
		  thrust::sort_by_key(anchors_rightmost_query_host+start, anchors_rightmost_query_host+end+1, anchors_leftmost_subject_host+start);
		  cudaMemcpy(anchors_rightmost_query_host, placeholder_anchors_rightmost_query_host, num_pvals_host*sizeof(int), cudaMemcpyHostToHost);
		  thrust::sort_by_key(anchors_rightmost_query_host+start, anchors_rightmost_query_host+end+1, anchors_rightmost_subject_host+start);
		  cudaStreamSynchronize(stream);		  
			  
		  same_anchor_left_query = anchors_leftmost_query_host[i];
		  start = i;
		  end = i;
		} else if(anchors_leftmost_query_host[i] == same_anchor_left_query){
		  end = i;
		} else{
		  same_anchor_left_query = anchors_leftmost_query_host[i];
		  start = i;
		  end = i;
		}
	  }
	  
	  start = 0;
	  end = 0;
	  int same_anchor_right_query = anchors_rightmost_query_host[0];
	  for(int i = 1; i < num_pvals_host; i++){
		int total_size = end - start;
		if((anchors_rightmost_query_host[i] != same_anchor_right_query || i == num_pvals_host-1) && total_size > 0){
		  if(i == num_pvals_host-1) end = i;
		  cudaMemcpy(placeholder_anchors_leftmost_subject_host, anchors_leftmost_subject_host, num_pvals_host*sizeof(long), cudaMemcpyHostToHost);
		  if(record_anchors){
		    thrust::sort_by_key(anchors_leftmost_subject_host+start, anchors_leftmost_subject_host+end+1, anchors_colinear_num_members_host+start);
	        cudaMemcpy(anchors_leftmost_subject_host, placeholder_anchors_leftmost_subject_host, num_pvals_host*sizeof(long), cudaMemcpyHostToHost);
	      }
		  thrust::sort_by_key(anchors_leftmost_subject_host+start, anchors_leftmost_subject_host+end+1, anchors_rightmost_subject_host+start);
		  cudaStreamSynchronize(stream);		  
			  
		  same_anchor_right_query = anchors_rightmost_query_host[i];
		  start = i;
		  end = i;
		} else if(anchors_rightmost_query_host[i] == same_anchor_right_query){
		  end = i;
		} else{
		  same_anchor_right_query = anchors_rightmost_query_host[i];
		  start = i;
		  end = i;
		}
	  }
	  
	  anchor_num_ranks = 1;
	  float min = anchor_pvals_host[0];
	  int count = 0;
	  for(int i = 0; i < num_pvals_host; i++) {
		  if(anchor_pvals_host[i] > min) {
			  min = anchor_pvals_host[i];
			  anchor_num_ranks += count;
			  count = 1;
		  } else {
			  count++;
		  }
		  anchor_pval_ranks_host[i] = anchor_num_ranks;
	  }
	  cudaMemcpy(anchor_pval_ranks, anchor_pval_ranks_host, sizeof(int)*num_pvals_host, cudaMemcpyHostToDevice);			CUERR("Copying anchor ranks back to GPU")
	  
      if(max_qvalue >= 0){
        cudaMalloc(&qvals, sizeof(float)*num_pvals_host);                                            CUERR("Allocating GPU memory for min DTW FDRs")
		fdr<<<fdr_griddim,threadblock_size,0,stream>>>(anchor_pvals, anchor_pval_ranks, num_pvals_host, qvals);    CUERR("Calculating false discovery rates");
      }
    }
    else{ 
      // Sort all the matches by p-values?  Maybe not... let's allow the caller to decide if they need to do that as otherwise we're just wasting time.
      // For example, they may only want the best match, or matches within a given subject location range, both of which are O(n) read-only rather than O(n log n) read-write.
    }


    // Package the results for return to the caller. For this to be anywhere close to efficient both timewise in allocation and GPU -> CPU copy later, 
    // we'll need to allocate a chunk of memory all at once to store the variable-length record match results (if requested withj arg record_anchors) in a way that 
    // can be nicely bulk transferred and unpacked by the client as a linked list sorted ascending by p-value.
    int total_sorted_match_records_size = num_pvals_host*sizeof(match_record);
    match_record *match_records_host;

    // Note that the following will be pinned memory (page locked), so client should free it quickly or risks affecting system 
    // performance on successive calls to this method.
    cudaMallocHost(&match_records_host, total_sorted_match_records_size);                 CUERR("Allocating CPU memory for match records");

    // Populate the results to be returned
    float *qvals_host = 0;
    int *anchor_pval_ranks_host = 0;
    int *anchor_pval_ordinals_host = 0;
    if(max_qvalue >= 0 || max_ranks > 0){
      cudaMallocHost(&qvals_host, sizeof(float)*num_pvals_host); 
      cudaMallocHost(&anchor_pval_ranks_host, sizeof(int)*num_pvals_host);                                            CUERR("Allocating CPU memory for DTW anchor p-value ranks");
      cudaMallocHost(&anchor_pval_ordinals_host, sizeof(int)*num_pvals_host);                                         CUERR("Allocating CPU memory for DTW anchor p-value ordinals");
      cudaMemcpyAsync(qvals_host, qvals, sizeof(float)*num_pvals_host, cudaMemcpyDeviceToHost, stream);                             CUERR("Copying FDRs passing criterion from GPU to CPU");
      cudaMemcpyAsync(anchor_pval_ranks_host, anchor_pval_ranks, sizeof(int)*num_pvals_host, cudaMemcpyDeviceToHost, stream);       CUERR("Copying FDR ranks passing criterion from GPU to CPU");
      cudaMemcpyAsync(anchor_pval_ordinals_host, anchor_pval_ordinals, sizeof(int)*num_pvals_host, cudaMemcpyDeviceToHost, stream); CUERR("Copying FDR ordinals passing criterion from GPU to CPU");
      unsigned int num_qvals_host;
      cudaMemcpyAsync(&num_qvals_host, num_kept_anchor_pvals, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream); CUERR("Copying number of FDRs passing criterion from GPU to CPU");
      cudaStreamSynchronize(stream);	CUERR("Synchronizing stream after copying FDRs from GPU to CPU");
	  
	  #if defined(_FLASH_UNIT_TEST)
	  std::ofstream fdr_file;
	  std::ofstream match_ord_file;
	  std::ofstream que_num_file;
	  std::ofstream pval_file;
	  std::ofstream anch_left_que_sorted_file;
	  std::ofstream anch_right_que_sorted_file;
	  std::ofstream anch_left_sub_sorted_file;
	  std::ofstream anch_right_sub_sorted_file;
	  
	  fdr_file.open(output_path + (std::string("fdr_test") + file_num).c_str());
	  match_ord_file.open(output_path + (std::string("match_ord_test") + file_num).c_str());
	  que_num_file.open(output_path + (std::string("que_num_test") + file_num).c_str());
	  pval_file.open(output_path + (std::string("pval_test") + file_num).c_str());
	  anch_left_que_sorted_file.open(output_path + (std::string("anch_left_que_sorted_test") + file_num).c_str());
	  anch_right_que_sorted_file.open(output_path + (std::string("anch_right_que_sorted_test") + file_num).c_str());
	  anch_left_sub_sorted_file.open(output_path + (std::string("anch_left_sub_sorted_test") + file_num).c_str());
	  anch_right_sub_sorted_file.open(output_path + (std::string("anch_right_sub_sorted_test") + file_num).c_str());
	  
	  for(int i = 0; i < num_qvals_host; i++){
	  	if((max_qvalue >= 0 && qvals_host[i] > max_qvalue) ||
	  		(max_ranks > 0 && anchor_pval_ranks_host[i] > max_ranks)){
	  		continue;
	  	}
	  	fdr_file << i << " - " << qvals_host[i] << std::endl;
	  	match_ord_file << i << " - " << anchor_pval_ranks_host[i] << std::endl;
	  	que_num_file << i << " - " << 0 << std::endl;
	  	pval_file << i << " - " << anchor_pvals_host[i] << std::endl;
	  	anch_left_que_sorted_file << i << " - " << anchors_leftmost_query_host[i] << std::endl;
	  	anch_right_que_sorted_file << i << " - " << anchors_rightmost_query_host[i] << std::endl;
	  	anch_left_sub_sorted_file << i << " - " << anchors_leftmost_subject_host[i] << std::endl;
	  	anch_right_sub_sorted_file << i << " - " << anchors_rightmost_subject_host[i] << std::endl;
	  }
	  std::cerr << std::endl;
	  
	  fdr_file.close();
	     match_ord_file.close();
	     que_num_file.close();
	     pval_file.close();
	  anch_left_que_sorted_file.close();
	  anch_right_que_sorted_file.close();
	  anch_left_sub_sorted_file.close();
	  anch_right_sub_sorted_file.close();
	#endif

      // Only record the subset that pass the criteria
      for(int i = 0; i < num_qvals_host; i++){
        if((max_qvalue >= 0 && qvals_host[i] > max_qvalue) ||
           (max_ranks > 0 && anchor_pval_ranks_host[i] > max_ranks)){
          continue;
        }
        match_records_host[i].fdr = qvals_host[i];
        match_records_host[i].match_ordinal = anchor_pval_ranks_host[i];
        match_records_host[i].query_number = 0; // TODO
        match_records_host[i].p_value = anchor_pvals_host[i];
        match_records_host[i].left_anchor_query = anchors_leftmost_query_host[i];
        match_records_host[i].right_anchor_query = anchors_rightmost_query_host[i];
        match_records_host[i].left_anchor_subject = anchors_leftmost_subject_host[i];
        match_records_host[i].right_anchor_subject = anchors_rightmost_subject_host[i];
        if(record_anchors) match_records_host[i].num_anchors = anchors_colinear_num_members_host[i];
        (*num_results)++;
      }
    }
    else{
      for(int i = 0; i < num_pvals_host; i++){
        match_records_host[i].query_number = 0; // TODO
        match_records_host[i].p_value = anchor_pvals_host[i];
        match_records_host[i].left_anchor_query = anchors_leftmost_query_host[i];
        match_records_host[i].right_anchor_query = anchors_rightmost_query_host[i];
        match_records_host[i].left_anchor_subject = anchors_leftmost_subject_host[i];
        match_records_host[i].right_anchor_subject = anchors_rightmost_subject_host[i];
        if(record_anchors) match_records_host[i].num_anchors = anchors_colinear_num_members_host[i];
      }
      (*num_results) = num_pvals_host;
    }
	(*results) = match_records_host;
    // We didn't actually store all the co-linear DTW anchors for each match below max_pvalue, just their count.
    // Now that we know the anchor count of all the results we plan on returning, we can allocate just enough 
    // memory to store them, and rerun the anchor calculation in storage mode (assuming the caller asked for them when invoking this method). 
    if(record_anchors){
      int total_num_anchors = 0;
      for(int i = 0; i < num_pvals_host; i++){
        total_num_anchors += anchors_colinear_num_members_host[i];
      }
      total_sorted_match_records_size += sizeof(match_record)*total_num_anchors;
      //TODO: rerun the anchor searches. Note that due to overlap in anchor search, actual non-redundant count will be lower than we allocated space for.
    }

    if(qvals)cudaFree(qvals);                                                                 CUERR("Freeing GPU memory for FDR values");
    if(anchor_pval_ranks)cudaFree(anchor_pval_ranks);                                         CUERR("Freeing GPU memory for p-value ranks");
    if(anchor_pval_ordinals)cudaFree(anchor_pval_ordinals);                                   CUERR("Freeing GPU memory for p-value ordinals");
    if(qvals_host)cudaFreeHost(qvals_host);                                                   CUERR("Freeing CPU memory for FDR values");
    if(anchor_pval_ranks_host)cudaFreeHost(anchor_pval_ranks_host);                           CUERR("Freeing CPU memory for p-value ranks");
    if(anchor_pval_ordinals_host)cudaFreeHost(anchor_pval_ordinals_host);                     CUERR("Freeing CPU memory for p-value ordinals");

  }

  if(anchor_pvals_host)cudaFreeHost(anchor_pvals_host);                                       CUERR("Freeing CPU memory for DTW anchors p-values");
  if(anchors_leftmost_query_host)cudaFreeHost(anchors_leftmost_query_host);                   CUERR("Freeing CPU memory for sets' first DTW query anchors");
  if(anchors_rightmost_query_host)cudaFreeHost(anchors_rightmost_query_host);                 CUERR("Freeing CPU memory for sets' last DTW query anchors");
  if(anchors_leftmost_subject_host)cudaFreeHost(anchors_leftmost_subject_host);               CUERR("Freeing CPU memory for sets' first DTW subject anchors");
  if(anchors_rightmost_subject_host)cudaFreeHost(anchors_rightmost_subject_host);             CUERR("Freeing CPU memory for sets' last DTW subject anchors");
  if(anchors_colinear_num_members_host)cudaFreeHost(anchors_colinear_num_members_host);       CUERR("Freeing CPU memory for set membership");
  if(anchors_colinear_num_members)cudaFree(anchors_colinear_num_members);                     CUERR("Freeing GPU memory for colinear anchor counts");
  cudaFree(anchor_pvals);                                                                     CUERR("Freeing GPU memory for DTW anchor p-values"); 
  cudaFree(anchors_leftmost_query);                                                           CUERR("Freeing GPU memory for sets' first DTW query anchors");
  cudaFree(anchors_rightmost_query);                                                          CUERR("Freeing GPU memory for sets' last DTW query anchors");
  cudaFree(anchors_leftmost_subject);                                                         CUERR("Freeing GPU memory for sets' first DTW subject anchors");
  cudaFree(anchors_rightmost_subject);                                                        CUERR("Freeing GPU memory for sets' last DTW subject anchors");

  return 0;
}
#endif
