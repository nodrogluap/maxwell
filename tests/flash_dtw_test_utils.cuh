#ifndef __FLASH_DTW_TEST_UTILS
#define __FLASH_DTW_TEST_UTILS

#include "../flash_dtw.cuh"

template <class T>
__global__ void set_membership_kernel(T* return_membership,  const int num_candidate_query_indices, long long* query_adjacent_candidates, const int minidtw_size, const float max_warp_proportion);

template <class T>
__global__ void calculate_pval_kernel(const float max_pval, int* query_adjacent_candidates, T* set_membership, QTYPE_ACC *sorted_colinear_distances, QTYPE_ACC *sorted_non_colinear_distances, 
									  int* num_sorted_non_colinear_distances_device, int* colinear_distance_lengths, 
									  unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, 
									  float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, 
									  int *output_num_members, bool func_used, bool all_test,
									  int* output_num_members_buff, float* output_pvals_buff, int* output_left_anchors_query_buff, 
									  int* output_right_anchors_query_buff, long* output_left_anchors_subject_buff, long* output_right_anchors_subject_buff);

template <class T>
__global__ void get_sorted_colinear_distances_kernel(QTYPE_ACC *sorted_colinear_distances, T* membership, QTYPE_ACC* query_adjacent_distances, int num_candidate_query_indices, int* num_sorted_colinear_distances);

template <class T>
__global__ void get_sorted_non_colinear_distances_kernel(QTYPE_ACC *sorted_non_colinear_distances, T* membership, const QTYPE_ACC *query_adjacent_distances, int* total_num_sorted_non_colinear_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, int* non_colinear_distance_lengths, int num_sorted_colinear_distances, bool thorough_calc);

__global__ void warpReduceSum_kernel(float* values, int values_length, float* return_value);

__global__ void warpReduceMin_kernel(float* values, int values_length, float* return_value);

__global__ void block_findMin_kernel(QTYPE* dtw_results, int dtw_dist_size, QTYPE_ACC* return_min, long long* return_index);

// Device side function that is used to call the inline function set_membership_results from flash_dtw.cuh
// return_membership - return buffer that will store the membership results from set_membership_results stored based on tid
// num_candidate_query_indices - the number of values to be storred in membership
// query_adjacent_candidates - indices of the subject that were found to be candidates
// minidtw_size - length of comparison between query and subject within a mini-DTW iteration
// max_warp_proportion- proportion of length deviation allowed between query and subject in alignment
template<class T>
__global__ void set_membership_kernel(T* return_membership,  const int num_candidate_query_indices, long long* query_adjacent_candidates, const int minidtw_size, const float max_warp_proportion){

	extern __shared__ char shared[];
	long long *subject_indices = (long long *) &shared[0];
	T *membership = (T *) &subject_indices[num_candidate_query_indices];
	

	if(threadIdx.x < num_candidate_query_indices) {
		subject_indices[threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
		membership[threadIdx.x] = -1;
	}
	
	set_membership_results<T>(membership, num_candidate_query_indices, subject_indices, minidtw_size, max_warp_proportion, true);

	return_membership[blockIdx.x*num_candidate_query_indices+threadIdx.x] = membership[threadIdx.x];

	// printf("num_tmp_members: %i, num_members: %i\n", num_tmp_members, *num_members);
	// __syncthreads();
}

// Device side function that is used to provide setup for and call get_sorted_colinear_distances from flash_dtw.cuh
// sorted_colinear_distances - buffer that will store the results from get_sorted_colinear_distances stored based on tid
// membership - memberships of the indeces that were passed into set_pairwise_bits stored based on tid
// query_adjacent_distances - distances calculated by soft_dtw / hard_dtw that will be sorted by get_sorted_colinear_distances
// num_candidate_query_indices - number of values to be stored in sorted_colinear_distances
template<class T>
__global__ void get_sorted_colinear_distances_kernel(QTYPE_ACC *sorted_colinear_distances, T* membership, QTYPE_ACC* query_adjacent_distances, int num_candidate_query_indices, int* num_sorted_colinear_distances){

	extern __shared__ char shared[];
	QTYPE_ACC *distances = (QTYPE_ACC *) &shared[0];
	QTYPE_ACC *my_thread_specific_sorted_colinear_distances = (QTYPE_ACC *) &distances[num_candidate_query_indices];

	if(threadIdx.x < num_candidate_query_indices) {
		distances[threadIdx.x] = query_adjacent_distances[threadIdx.x+blockIdx.x*num_candidate_query_indices];
	}
	__syncthreads();

	int num_members = 0;
	for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){ // start at threadIdx.x because we know that the leftmost member of the disjoint set being examined is the thread being executed
		if(membership[i] == threadIdx.x){
			num_members++;
		}
	}
	__syncthreads();

	int *partition_counter = (int *) &my_thread_specific_sorted_colinear_distances[num_candidate_query_indices+(num_candidate_query_indices%2)];
	if(threadIdx.x == 0) {
		(*partition_counter) = 0;
	}
	__syncthreads();
	
	if(num_members == 0){
		return;
	}
	
	int local_partition = 0;

	get_sorted_colinear_distances<T>(my_thread_specific_sorted_colinear_distances, membership, distances, num_candidate_query_indices, partition_counter, &local_partition, num_members, num_sorted_colinear_distances, true);

	int num_dists_stored = 0;
	for(int i = blockIdx.x*num_candidate_query_indices; i < (blockIdx.x+1)*num_candidate_query_indices; ++i){ 
		sorted_colinear_distances[i] = my_thread_specific_sorted_colinear_distances[num_dists_stored];
		num_dists_stored++;
	}
}

// TODO: Structure how sorted_non_colinear_distances based on sorted_colinear_distances above as there will probably be future problems occuring here.
// Device side function that is used to call get_sorted_non_colinear_distances from flash_dtw.cuh
// sorted_non_colinear_distances - buffer that will store the results from get_sorted_non_colinear_distances stored based on tid
// total_num_sorted_non_colinear_distances - the total number of non colinear distances that get_sorted_non_colinear_distances returns
// set_membership - memberships of the indeces that were passed into set_pairwise_bits stored based on tid
// dtw_distances - distances calculated by soft_dtw / hard_dtw that will be sorted by get_sorted_colinear_distances
// num_dtw_distances - size of dtw_distances
// query_adjacent_candidates - the query candidates returned from set_pairwise_bits stored based on tid
// non_colinear_distance_lengths - the lengths of each non colinear distances that relate to each member stored based on tid
template <class T>
__global__ void get_sorted_non_colinear_distances_kernel(QTYPE_ACC *sorted_non_colinear_distances, T* membership, const QTYPE_ACC *query_adjacent_distances, int* total_num_sorted_non_colinear_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, int* non_colinear_distance_lengths, int num_sorted_colinear_distances, bool thorough_calc){

	// extern __shared__ char shared[];
	QTYPE_ACC *my_thread_specific_sorted_non_colinear_distances = (QTYPE_ACC *) &sorted_non_colinear_distances[CUDA_THREADBLOCK_MAX_THREADS*(threadIdx.x%num_candidate_query_indices)+CUDA_THREADBLOCK_MAX_THREADS*num_candidate_query_indices*blockIdx.x];
	int bid = blockIdx.x*blockDim.x;

	int num_sorted_non_colinear_distances = 0;
	int num_members = 0;
	for(int i = threadIdx.x; i < num_candidate_query_indices; ++i){ // start at threadIdx.x because we know that the leftmost member of the disjoint set being examined is the thread being executed
		if(membership[i] == threadIdx.x){
			num_members++;
		}
	}
	__syncthreads();

	if(num_members == 0){
		return;
	}

	if(thorough_calc){
		get_sorted_non_colinear_distances<T>(my_thread_specific_sorted_non_colinear_distances, membership, num_members, query_adjacent_distances, &num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, thorough_calc, 0, num_sorted_colinear_distances, false);
		// for(int j = 0; j < num_sorted_non_colinear_distances; j++){
			// sorted_non_colinear_distances[blockIdx.x*blockDim.x+j] = my_thread_specific_sorted_non_colinear_distances[j];
		// }
		(*total_num_sorted_non_colinear_distances) += num_sorted_non_colinear_distances;
		non_colinear_distance_lengths[bid] = num_sorted_non_colinear_distances;
	} else{
		get_sorted_non_colinear_distances<T>(my_thread_specific_sorted_non_colinear_distances, membership, num_members, query_adjacent_distances, &num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, thorough_calc, 0, num_sorted_colinear_distances, false);
		// for(int j = 0; j < num_sorted_non_colinear_distances; j++){
			// sorted_non_colinear_distances[blockIdx.x*blockDim.x+j] = my_thread_specific_sorted_non_colinear_distances[j];
		// }
		(*total_num_sorted_non_colinear_distances) += num_sorted_non_colinear_distances;
		non_colinear_distance_lengths[bid] = num_sorted_non_colinear_distances;
	}
}

// Device side function that is used to call get_sorted_non_colinear_distances from flash_dtw.cuh
// max_pval - the maximum pvalue used to accept or reject a match. If a pvalue is found to be lower than this value, then it wont be counted
// query_adjacent_candidates - the query candidates returned from set_pairwise_bits
// set_membership - memberships of the indeces that were passed into set_pairwise_bits stored based on tid
// sorted_colinear_distances- colinear distances found for each set of memberships that are sorted in acending order stored based on tid
// sorted_non_colinear_distances - non colinear distances found that are sorted in acending order stored based on tid
// num_sorted_non_colinear_distances_device - the number of non colinear distances
// colinear_distance_lengths - the length of each sorted_colinear_distance  stored based on tid
// num_results_recorded - the total number of results that are found to be under the maximum pvalue by calculate_pval
// num_results_notrecorded - the total number of results that are found to be over the maximum by calculate_pval
// max_num_results - the maximum number of results that can be recorded
// output_pvals - the buffer that will store all pvalues based on its tid
// output_left_anchors_query - the buffer that will store all left anchors in the query based on its tid
// output_right_anchors_query - the buffer that will store all right anchors in the query based on its tid
// output_left_anchors_subject - the buffer that will store all left anchors in the subject based on its tid
// output_right_anchors_subject - the buffer that will store all right anchors in the subject based on its tid
// output_num_members - the buffer that will store the number of members that are looked at for each pvalue based on its tid
// first_idxs - the first idx in candidate_subject_indices. Also the value that is stored in return_query_candidates
// func_used - flag that checks how sorted_non_colinear_distances was populated. Is true if get_min_distance_sorted_subject_sample was used. Is false if get_sorted_non_colinear_distances was used
// output_num_members_buff - Additional buffer used for testing that stores the number of members that are looked at for each pvalue based on its tid
// output_pvals_buff - Additional buffer used for testing that stores all pvalues based on its tid
// output_left_anchors_query_buff - Additional buffer used for testing that stores all left anchors in the query based on its tid
// output_right_anchors_query_buff - Additional buffer used for testing that stores all right anchors in the query based on its tid
// output_left_anchors_subject_buff - Additional buffer used for testing that stores all left anchors in the subject based on its tid
// output_right_anchors_subject_buff - Additional buffer used for testing that stores all right anchors in the subject based on its tid

template <class T>
__global__ void calculate_pval_kernel(const float max_pval, int* query_adjacent_candidates, T* membership, QTYPE_ACC *sorted_colinear_distances, QTYPE_ACC *sorted_non_colinear_distances, 
									  int* num_sorted_non_colinear_distances_device, int* colinear_distance_lengths, 
									  unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, 
									  float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, long *output_left_anchors_subject, long *output_right_anchors_subject, 
									  int *output_num_members, bool func_used, bool all_test,
									  int* output_num_members_buff, float* output_pvals_buff, int* output_left_anchors_query_buff, 
									  int* output_right_anchors_query_buff, long* output_left_anchors_subject_buff, long* output_right_anchors_subject_buff, int num_candidate_query_indices, int minidtw_size){

	// QTYPE_ACC my_thread_specific_sorted_colinear_distances[256];

	extern __shared__ char shared[];
	long long *subject_indices = (long long *) &shared[0];

	if(threadIdx.x < num_candidate_query_indices) {
		subject_indices[threadIdx.x] = query_adjacent_candidates[threadIdx.x+blockIdx.x*num_candidate_query_indices];
	}
	__syncthreads();

	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	int bid = blockIdx.x*blockDim.x;

	if(blockIdx.x == 0 && threadIdx.x == 0){
		// Make sure num_results_recorded and num_results_notrecorded are intialized
		atomicExch(num_results_recorded, 0);
		atomicExch(num_results_notrecorded, 0);
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

	if(num_members == 0){
		return;
	}

	calculate_pval(max_pval, sorted_colinear_distances, &sorted_non_colinear_distances[func_used? 0 : bid], num_sorted_non_colinear_distances_device[all_test? bid : 0], num_results_recorded, num_results_notrecorded, max_num_results, output_pvals, leftmost_anchor_query, rightmost_anchor_query, leftmost_anchor_subject, rightmost_anchor_subject, all_test ? colinear_distance_lengths[tid] : num_members, output_left_anchors_query, output_right_anchors_query, output_left_anchors_subject, output_right_anchors_subject, output_num_members, output_num_members_buff, output_pvals_buff, output_left_anchors_query_buff, output_right_anchors_query_buff, output_left_anchors_subject_buff, output_right_anchors_subject_buff);

}

__global__ void warpReduceSum_kernel(float* values, int values_length, float* return_value){
	int pos = blockIdx.x*blockDim.x+threadIdx.x;
	float val_here = 0;

	if(pos < values_length){
		val_here = values[pos];
	}
	 __syncwarp();

	val_here = warpReduceSum(val_here);

	__syncwarp();
	__syncthreads();
	if(pos == 0){
		*return_value = val_here;
	}
}

__global__ void warpReduceMin_kernel(float* values, int values_length, float* return_value){
	int pos = blockIdx.x*blockDim.x+threadIdx.x;
	float val_here = 0;

	if(pos < values_length){
		val_here = values[pos];
	}
	 __syncwarp();

	val_here = warpReduceMin(val_here);

	__syncwarp();
	__syncthreads();
	if(pos == 0){
		*return_value = val_here;
	}
}

__global__ void block_findMin_kernel(QTYPE* dtw_results, int dtw_dist_size, QTYPE_ACC* return_min, long long* return_index){
	long long jobid = (long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x;
	long long strided_jobid = jobid*MINIDTW_STRIDE;
	const int block_dim = CUDA_THREADBLOCK_MAX_THREADS / MINIDTW_STRIDE;

	__shared__ char shared[sizeof(long long)*block_dim/2];
	
	long long *idata = (long long *) &shared[0];

	QTYPE_ACC min = DTW_MAX;
	long long index = -1;
	
	block_findMin(&dtw_results[blockIdx.x*dtw_dist_size], idata, threadIdx.x, blockDim.x, strided_jobid, min, index);

	if(threadIdx.x == 0){
		return_min[blockIdx.x] = min;
		return_index[blockIdx.x] = index;
	}
	__syncthreads();
}

#endif