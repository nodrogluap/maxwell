#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdlib.h>     /* srand, rand */
#include <string>

#include <iostream>
#include <fstream>

#if defined(_WIN32)
	#include <direct.h>
	#include <conio.h>
	#include <windows.h>
	#include <bitset>
	extern "C"{
		#include "getopt.h"
	}
	#define GetCurrentDir _getcwd
	#define ONWINDOWS 1
#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#define ONWINDOWS 0
#endif

#include "../cuda_utils.h" // CUERR() and timer functions
#include "../flash_utils.hpp" // Needed for reading in binary data

#define QTYPE_float
#define QTYPE_ACC_float

#ifndef MINIDTW_STRIDE
#define MINIDTW_STRIDE 1
#endif

// The search algorithm 
#include "test_utils.cuh"
#include "flash_dtw_test_utils.cuh"
#include "../flash_dtw_utils.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

// void get_znorm_stats(QTYPE *data, long data_length, float *mean, float *stddev, QTYPE *min, QTYPE *max, cudaStream_t stream=0);
TEST_CASE( " Get Znorm Stats " ){

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	QTYPE *good_subject_values;
	unsigned long long int num_subject_values;
	int result = read_binary_data<QTYPE>(good_file.c_str(), &good_subject_values, &num_subject_values);

	float *mean;
	float *stddev;
	QTYPE *min;
	QTYPE *max;

	cudaMalloc(&mean, sizeof(float)); CUERR("Allocating GPU memory for query mean");
    cudaMalloc(&stddev, sizeof(float)); CUERR("Allocating GPU memory for query std dev");
    cudaMalloc(&min, sizeof(QTYPE)); CUERR("Allocating GPU memory for query min");
    cudaMalloc(&max, sizeof(QTYPE)); CUERR("Allocating GPU memory for query max");
	
	SECTION("Good Data"){
		std::cerr << "------TEST ZNORM_STATS GOOD DATA------" << std::endl;

		// std::cerr << "Subject : ";
		// for(int i = 0; i < num_subject_values; i++){
			// std::cerr << good_subject_values[i] << ", ";
		// }
		// std::cerr << std::endl;

		QTYPE *data;
		cudaMalloc(&data, sizeof(QTYPE)*num_subject_values);		CUERR("Allocating data");

		cudaMemcpy(data, good_subject_values, sizeof(QTYPE)*num_subject_values, cudaMemcpyHostToDevice);			CUERR("Copying data");
		
		get_znorm_stats(data, num_subject_values, mean, stddev, min, max);

		float return_mean;
		float return_stddev;
		QTYPE return_min;
		QTYPE return_max;

		cudaMemcpy(&return_mean, mean, sizeof(float), cudaMemcpyDeviceToHost);		CUERR("Copying mean");
		cudaMemcpy(&return_stddev, stddev, sizeof(float), cudaMemcpyDeviceToHost);	CUERR("Copying stddev");
		cudaMemcpy(&return_min, min, sizeof(QTYPE), cudaMemcpyDeviceToHost);		CUERR("Copying min");
		cudaMemcpy(&return_max, max, sizeof(QTYPE), cudaMemcpyDeviceToHost);		CUERR("Copying max");

		REQUIRE( return_mean == 880.77161f );
		REQUIRE( return_stddev == 146.36993f );
		REQUIRE( return_min == 592 );
		REQUIRE( return_max == 1274 );

		cudaFree(data);
		
		std::cerr << std::endl;
	}

	cudaFree(mean);		CUERR("free mean");
	cudaFree(stddev);   CUERR("free stddev");
	cudaFree(min);      CUERR("free min");
	cudaFree(max);      CUERR("free max");
}

// void load_subject(QTYPE *subject, QTYPE *subject_std, long subject_length, int use_std, cudaStream_t stream)
TEST_CASE( " Load Subject " ) {

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	QTYPE *good_subject_values;
	unsigned long long int num_subject_values;
	int result = read_binary_data<QTYPE>(good_file.c_str(), &good_subject_values, &num_subject_values);

	SECTION("Good Subject No STD"){
		std::cerr << "------LOAD_SUBJECT GOOD SUB------" << std::endl;
		load_subject(good_subject_values, 0, num_subject_values, false);

		QTYPE* return_subject;
		long return_length;
		float return_mean;
		float return_stddev;
		QTYPE return_min;
		QTYPE return_max;
		
		get_subject_from_GPU(&return_subject, &return_length, &return_mean, &return_stddev, &return_min, &return_max);

		REQUIRE( return_subject[0] == 849 );
		REQUIRE( return_length == 479 );
		REQUIRE( return_subject[478] == 711 );

		REQUIRE( return_mean == 880.77161f );
		REQUIRE( return_stddev == 146.36993f );
		REQUIRE( return_min == 592 );
		REQUIRE( return_max == 1274 );

		free(return_subject);
		std::cerr << std::endl;
	}

}

// void mean_min_max(QTYPE *data, int data_length, float *threadblock_means, QTYPE *threadblock_mins, QTYPE *threadblock_maxs);
TEST_CASE( " Mean Min Max Kernel " ){

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	QTYPE *good_subject_values;
	unsigned long long int num_subject_values;
	int result = read_binary_data<QTYPE>(good_file.c_str(), &good_subject_values, &num_subject_values);

	SECTION("Good data"){
		std::cerr << "------TEST MIN_MAX GOOD DATA------" << std::endl;

		int num_threadblocks = DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS);
		float *threadblock_means;
		QTYPE *threadblock_mins;
		QTYPE *threadblock_maxs;

		cudaMalloc(&threadblock_means, sizeof(float)*num_threadblocks);               CUERR("Allocating device memory for query Z-norm threadblock means");
		cudaMalloc(&threadblock_mins, sizeof(QTYPE)*num_threadblocks);                CUERR("Allocating device memory for query Z-norm threadblock mins");
		cudaMalloc(&threadblock_maxs, sizeof(QTYPE)*num_threadblocks);                CUERR("Allocating device memory for query Z-norm threadblock maxs");

		QTYPE *data;
		cudaMalloc(&data, sizeof(QTYPE)*num_subject_values);		CUERR("Allocating data");

		cudaMemcpy(data, good_subject_values, sizeof(QTYPE)*num_subject_values, cudaMemcpyHostToDevice);			CUERR("Copying data");
		
		dim3 grid(num_threadblocks, 1, 1);
		int req_threadblock_shared_memory = CUDA_THREADBLOCK_MAX_THREADS*sizeof(QTYPE)+(sizeof(QTYPE)*2+sizeof(float)+2)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
		mean_min_max<<<grid,CUDA_THREADBLOCK_MAX_THREADS,req_threadblock_shared_memory,0>>>(data, num_subject_values, threadblock_means, threadblock_mins, threadblock_maxs); CUERR("Calculating data mean/min/max");
 

		float* return_mean = (float*) malloc(sizeof(float)*num_threadblocks);
		QTYPE* return_min = (QTYPE*) malloc(sizeof(float)*num_threadblocks);
		QTYPE* return_max = (QTYPE*) malloc(sizeof(float)*num_threadblocks);

		cudaMemcpy(return_mean, threadblock_means, sizeof(float)*num_threadblocks, cudaMemcpyDeviceToHost);		CUERR("Copying mean");
		cudaMemcpy(return_min, threadblock_mins, sizeof(QTYPE)*num_threadblocks, cudaMemcpyDeviceToHost);		CUERR("Copying min");
		cudaMemcpy(return_max, threadblock_maxs, sizeof(QTYPE)*num_threadblocks, cudaMemcpyDeviceToHost);		CUERR("Copying max");

		REQUIRE( return_mean[0] == 893.48047f );
		REQUIRE( return_min[0] == 592 );
		REQUIRE( return_max[0] == 1249 );

		cudaFree(threadblock_means);                                                CUERR("Freeing device memory for query Z-norm threadblock means");
		cudaFree(threadblock_mins);                                                 CUERR("Freeing device memory for query Z-norm threadblock mins");
		cudaFree(threadblock_maxs);                                                 CUERR("Freeing device memory for query Z-norm threadblock maxs");
		cudaFree(data);		CUERR("Free data");

		free(return_mean);
		free(return_min);
		free(return_max);
		
		std::cerr << std::endl;
	}

}

// void hard_dtw(const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, int minidtw_size)
TEST_CASE(" Hard DTW "){

	int subject_length = 10;
	QTYPE subject[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	int query_length = 10;

	int minidtw_size = 10;

	long num_query_indices = query_length/minidtw_size;

	size_t mem_size = DIV_ROUNDUP(subject_length, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices;

	QTYPE_ACC *query_adjacent_distances = NULL;
	cudaMalloc(&query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size);                              CUERR("Allocating GPU memory for DTW anchor distances")

	long long *query_adjacent_candidates = NULL;
	cudaMalloc(&query_adjacent_candidates, sizeof(long long)*mem_size);                              CUERR("Allocating GPU memory for DTW anchor distances")

	int threadblock_size_dtw = 10; 
	dim3 griddim_dtw(DIV_ROUNDUP(subject_length, threadblock_size_dtw)*num_query_indices, 1, 1);

	SECTION(" Good Data Matching Sub and Query"){
		std::cerr << "------TEST HARD_DTW GOOD DATA SAME SUB/ QUERY------" << std::endl;

		QTYPE query[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

		load_subject(subject, 0, subject_length, false);
		flash_dtw_setup(query, query_length, NO_ZNORM);

		// std::cerr << "num_query_indices: " << num_query_indices << std::endl;
		// std::cerr << "griddim.x: " << griddim.x << ", threadblock_size: " << threadblock_size << std::endl;

		hard_dtw<<<griddim_dtw, threadblock_size_dtw, 0, 0>>>(num_query_indices, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates, query_adjacent_distances, minidtw_size); CUERR("Running DTW anchor distance calculations")

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size);
		cudaMemcpy(return_distances, query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size, cudaMemcpyDeviceToHost);		CUERR("Copy distances from device to host");

		long long* return_candidates = (long long*) malloc(sizeof(long long)*mem_size);
		cudaMemcpy(return_candidates, query_adjacent_candidates, sizeof(long long)*mem_size, cudaMemcpyDeviceToHost);		CUERR("Copy candidates from device to host");

		// std::cerr << "Candidates: ";
		// for(int i = 0; i < mem_size; i++){
			// std::cerr << return_candidates[i] << ", " << std::endl;
		// }
		// std::cerr << std::endl;

		// std::cerr << "Distances: ";
		// for(int i = 0; i < mem_size; i++){
			// std::cerr << return_distances[i] << ", " << std::endl;
		// }

		REQUIRE( return_distances[0] == 0 );

		free(return_distances);
		free(return_candidates);
		std::cerr << std::endl;
	}

	SECTION(" Good Data Different Sub and Query"){
		std::cerr << "------TEST HARD_DTW GOOD DATA DIFF SUB/ QUERY------" << std::endl;

		QTYPE query[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

		load_subject(subject, 0, subject_length, false);
		flash_dtw_setup(query, query_length, NO_ZNORM);

		// std::cerr << "num_query_indices: " << num_query_indices << std::endl;
		// std::cerr << "griddim.x: " << griddim.x << ", threadblock_size: " << threadblock_size << std::endl;

		hard_dtw<<<griddim_dtw, threadblock_size_dtw, 0, 0>>>(num_query_indices, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates, query_adjacent_distances, minidtw_size); CUERR("Running DTW anchor distance calculations")

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size);
		cudaMemcpy(return_distances, query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size, cudaMemcpyDeviceToHost);	CUERR("Copy distances from device to host");

		long long* return_candidates = (long long*) malloc(sizeof(long long)*mem_size);
		cudaMemcpy(return_candidates, query_adjacent_candidates, sizeof(long long)*mem_size, cudaMemcpyDeviceToHost);		CUERR("Copy candidates from device to host");

		// std::cerr << "Candidates: ";
		// for(int i = 0; i < mem_size; i++){
			// std::cerr << return_candidates[i] << ", " << std::endl;
		// }
		// std::cerr << std::endl;

		// std::cerr << "Distances: ";
		// for(int i = 0; i < mem_size; i++){
			// std::cerr << return_distances[i] << ", " << std::endl;
		// }

		REQUIRE( return_distances[0] == 2 );

		free(return_distances);
		free(return_candidates);
		std::cerr << std::endl;
	}

	SECTION(" Good Data Matching Sub and Query"){
		std::cerr << "------TEST HARD_DTW LONG SUB------" << std::endl;

		int subject_length_long = 20;
		QTYPE subject_long[] = {11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20};

		QTYPE query[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

		load_subject(subject_long, 0, subject_length_long, false);
		flash_dtw_setup(query, query_length, NO_ZNORM);

		// std::cerr << "num_query_indices_long: " << num_query_indices_long << std::endl;
		// std::cerr << "griddim.x: " << griddim.x << ", threadblock_size: " << threadblock_size << std::endl;

		hard_dtw<<<griddim_dtw, threadblock_size_dtw, 0, 0>>>(num_query_indices, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates, query_adjacent_distances, minidtw_size); CUERR("Running DTW anchor distance calculations")

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size);
		cudaMemcpy(return_distances, query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size, cudaMemcpyDeviceToHost);	CUERR("Copy distances from device to host");

		long long* return_candidates = (long long*) malloc(sizeof(long long)*mem_size);
		cudaMemcpy(return_candidates, query_adjacent_candidates, sizeof(long long)*mem_size, cudaMemcpyDeviceToHost);		CUERR("Copy candidates from device to host");

		// std::cerr << "Distances: ";
		// for(int i = 0; i < mem_size; i++){
			// std::cerr << return_distances[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE( (int)return_distances[3] == 0 );

		free(return_distances);
		free(return_candidates);
		std::cerr << std::endl;
	}

	SECTION("Test Files"){

		std::cerr << "------TEST HARD_DTW WITH TEST FILES------" << std::endl;

		std::string good_file_sub = current_working_dir + "/good_files/text/good_test_sub.txt";
		std::string good_file_query = current_working_dir + "/good_files/text/good_test_query.txt";
		QTYPE *good_subject_values;
		unsigned long long int num_subject_values;
		int result_test = read_text_data<QTYPE>(good_file_sub.c_str(), &good_subject_values, &num_subject_values);

		// std::cerr << "Subject: ";
		// for(int i = 0; i < num_subject_values; i++){
			// std::cerr << good_subject_values[i] << ", ";
		// }
		// std::cerr << std::endl << std::endl;

		QTYPE *good_query_values;
		unsigned long long int num_query_values;
		int result = read_text_data<QTYPE>(good_file_query.c_str(), &good_query_values, &num_query_values);

		// std::cerr << "Query: ";
		// for(int i = 0; i < num_query_values; i++){
			// std::cerr << good_query_values[i] << ", ";
		// }
		// std::cerr << std::endl << std::endl;

		long num_query_indices_test = num_query_values/minidtw_size;
		size_t mem_size_test = DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices_test;
		QTYPE_ACC *query_adjacent_distances_test = NULL;
		cudaMalloc(&query_adjacent_distances_test, sizeof(QTYPE_ACC)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")

		long long *query_adjacent_candidates_test = NULL;
		cudaMalloc(&query_adjacent_candidates_test, sizeof(long long)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")
	
		dim3 griddim_dtw_test(DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices_test, 1, 1);

		load_subject(good_subject_values, 0, num_subject_values, false);

		flash_dtw_setup(good_query_values, num_query_values, NO_ZNORM);

		std::cerr << "Running hard_dtw (" << griddim_dtw_test.x << ", " << griddim_dtw_test.y << ", " << griddim_dtw_test.z << ")" << std::endl;
		int threadblock_size_dtw_test = CUDA_THREADBLOCK_MAX_THREADS;
		hard_dtw<<<griddim_dtw_test, threadblock_size_dtw_test, 0, 0>>>(num_query_indices_test, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates_test, query_adjacent_distances_test, minidtw_size); CUERR("Running DTW anchor distance calculations")

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size_test);
		cudaMemcpy(return_distances, query_adjacent_distances_test, sizeof(QTYPE_ACC)*mem_size_test, cudaMemcpyDeviceToHost);	CUERR("Copy distances from device to host");

		long long* return_candidates = (long long*) malloc(sizeof(long long)*mem_size_test);
		cudaMemcpy(return_candidates, query_adjacent_candidates_test, sizeof(long long)*mem_size_test, cudaMemcpyDeviceToHost);		CUERR("Copy candidates from device to host");

		// std::cerr << "num_subject_values: " << num_subject_values << ", num_query_values: " << num_query_values << std::endl;

		std::cerr << "Distances: ";
		for(int i = 0; i < mem_size_test; i++){
			if(return_distances[i] == 0){
				std::cerr << return_distances[i] << ", " <<  return_candidates[i] << " - " << i << ", ";
			}
		}
		std::cerr << std::endl;

		REQUIRE( return_distances[20] == 0 );
		REQUIRE( return_candidates[20] == 400 );

		free(return_distances);
		free(return_candidates);
		cudaFree(query_adjacent_distances_test);		CUERR("Free long distances");
		cudaFree(query_adjacent_candidates_test);		CUERR("Free long candidates");


		std::cerr << std::endl;
	}

	SECTION("Rscript Generated Files"){
		std::cerr << "------TEST HARD_DTW WITH RSCRIPT GENERATED TEST FILES------" << std::endl;

		std::string good_file_sub = current_working_dir + "/good_files/text/dtw_l2_norm_subject.txt";
		std::string good_file_query = current_working_dir + "/good_files/text/dtw_l2_norm_query.txt";
		QTYPE *good_subject_values;
		unsigned long long int num_subject_values;
		int result_test = read_text_data<QTYPE>(good_file_sub.c_str(), &good_subject_values, &num_subject_values);

		REQUIRE( num_subject_values == 10 );

		QTYPE *good_query_values;
		unsigned long long int num_query_values;
		int result = read_text_data<QTYPE>(good_file_query.c_str(), &good_query_values, &num_query_values);

		REQUIRE( num_query_values == 10 );

		long num_query_indices_test = num_query_values/minidtw_size;
		size_t mem_size_test = DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices_test;
		QTYPE_ACC *query_adjacent_distances_test = NULL;
		cudaMalloc(&query_adjacent_distances_test, sizeof(QTYPE_ACC)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")

		long long *query_adjacent_candidates_test = NULL;
		cudaMalloc(&query_adjacent_candidates_test, sizeof(long long)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")
	
		dim3 griddim_dtw_test(DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices_test, 1, 1);

		load_subject(good_subject_values, 0, num_subject_values, false);
		flash_dtw_setup(good_query_values, num_query_values, NO_ZNORM);

		// std::cerr << "num_query_indices_long: " << num_query_indices_long << std::endl;
		// std::cerr << "griddim.x: " << griddim.x << ", threadblock_size: " << threadblock_size << std::endl;

		hard_dtw<<<griddim_dtw_test, threadblock_size_dtw, 0, 0>>>(num_query_indices_test, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates_test, query_adjacent_distances_test, minidtw_size); CUERR("Running DTW anchor distance calculations")

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size_test);
		cudaMemcpy(return_distances, query_adjacent_distances_test, sizeof(QTYPE_ACC)*mem_size_test, cudaMemcpyDeviceToHost);	CUERR("Copy distances from device to host");

		long long* return_candidates = (long long*) malloc(sizeof(long long)*mem_size);
		cudaMemcpy(return_candidates, query_adjacent_candidates, sizeof(long long)*mem_size, cudaMemcpyDeviceToHost);		CUERR("Copy candidates from device to host");

		// std::cerr << "num_subject_values: " << num_subject_values << ", num_query_values: " << num_query_values << std::endl;

		// std::cerr << "Distances: ";
		// for(int i = 0; i < mem_size_test; i++){
			// std::cerr << return_distances[i] << " - " << i << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE( return_distances[0] == 4313 );

		free(return_distances);
		free(return_candidates);
		cudaFree(query_adjacent_distances_test);		CUERR("Free long distances");
		cudaFree(query_adjacent_candidates_test);		CUERR("Free long candidates");

		std::cerr << std::endl;
	}

	cudaFree(query_adjacent_distances);	CUERR("free distances");
	cudaFree(query_adjacent_candidates);	CUERR("free candidates");
}

// void soft_dtw_wrap(const dim3 griddim_dtw, const int threadblock_size_dtw, cudaStream_t stream, const int num_query_indices, QTYPE * subject, QTYPE * subject_std, long long * query_adjacent_candidates, QTYPE_ACC * query_adjacent_distances, const int minidtw_size, const int minidtw_warp, const int use_std)
TEST_CASE(" Soft DTW "){

	int minidtw_size = 10;
	int minidtw_warp = 2;

	SECTION("Rscript Generated Files"){
		std::cerr << "------TEST SOFT_DTW WITH RSCRIPT GENERATED TEST FILES------" << std::endl;

		std::string good_file_sub = current_working_dir + "/good_files/text/dtw_l2_norm_subject.txt";
		std::string good_file_query = current_working_dir + "/good_files/text/dtw_l2_norm_query.txt";
		QTYPE *good_subject_values;
		unsigned long long int num_subject_values;
		int result_test = read_text_data<QTYPE>(good_file_sub.c_str(), &good_subject_values, &num_subject_values);

		REQUIRE( num_subject_values == 10 );

		QTYPE *good_query_values;
		unsigned long long int num_query_values;
		int result = read_text_data<QTYPE>(good_file_query.c_str(), &good_query_values, &num_query_values);

		REQUIRE( num_query_values == 10 );

		long num_query_indices = num_query_values/minidtw_size;
	
		int threadblock_size_dtw = CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE; 
		dim3 griddim_dtw(DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices, 1, 1);

		size_t mem_size_test = DIV_ROUNDUP(num_subject_values, CUDA_THREADBLOCK_MAX_THREADS)*num_query_indices;
		QTYPE_ACC *query_adjacent_distances = NULL;
		cudaMalloc(&query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")

		long long *query_adjacent_candidates = NULL;
		cudaMalloc(&query_adjacent_candidates, sizeof(long long)*mem_size_test);                              CUERR("Allocating GPU memory for DTW anchor distances")

		load_subject(good_subject_values, 0, num_subject_values, false);
		flash_dtw_setup(good_query_values, num_query_values, NO_ZNORM);

		// std::cerr << "num_query_indices_long: " << num_query_indices_long << std::endl;
		// std::cerr << "griddim.x: " << griddim.x << ", threadblock_size: " << threadblock_size << std::endl;

		soft_dtw_wrap(griddim_dtw, threadblock_size_dtw, 0, num_query_indices, get_subject_pointer(), get_subject_std_pointer(), query_adjacent_candidates, query_adjacent_distances, minidtw_size, minidtw_warp, false);

		QTYPE_ACC* return_distances = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*mem_size_test);
		cudaMemcpy(return_distances, query_adjacent_distances, sizeof(QTYPE_ACC)*mem_size_test, cudaMemcpyDeviceToHost);	CUERR("Copy distances from device to host");

		// std::cerr << "num_subject_values: " << num_subject_values << ", num_query_values: " << num_query_values << std::endl;

		// std::cerr << "Distances: ";
		// for(int i = 0; i < mem_size_test; i++){
			// std::cerr << return_distances[i] << " - " << i << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE( return_distances[0] == 4313 );

		free(return_distances);

		cudaFree(query_adjacent_distances);		CUERR("Free long distances");
		cudaFree(query_adjacent_candidates);		CUERR("Free long candidates");


		std::cerr << std::endl;
	}
}

// void set_membership_kernel(T* return_membership,  const int num_candidate_query_indices, long long* query_adjacent_candidates, const int minidtw_size, const float max_warp_proportion)
TEST_CASE(" SET MEMBERSHIP "){	

	int minidtw_size = 10;
	float max_warp_proportion = 0.25;
	long num_query_indices = 100;

	SECTION(" Good Data "){

		std::cerr << "------TEST SET MEMBERSHIP GOOD DATA------" << std::endl;

		short* set_membership;
		cudaMalloc(&set_membership, sizeof(short)*num_query_indices);  	CUERR("Allocate memory for membership"); 

		long long* QAC_host = (long long*)malloc(sizeof(long long)*num_query_indices);
		

		for(int i = 0; i < num_query_indices; i++){
			if(i%2 == 0)
				QAC_host[i] = i*10;
			else
				QAC_host[i] = 1;
		}

		long long* query_adjacent_candidates;
		cudaMalloc(&query_adjacent_candidates, sizeof(long long)*num_query_indices);  	CUERR("Allocate memory for candidates");
		cudaMemcpy(query_adjacent_candidates, QAC_host,  sizeof(long long)*num_query_indices, cudaMemcpyHostToDevice);
		
		int threadblock_size = int(num_query_indices);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual

		unsigned int *num_members = 0;
		cudaMalloc((void**)&num_members, sizeof(unsigned int));		CUERR("Allocate num members");

		size_t required_threadblock_shared_memory = 100*(sizeof(short)+sizeof(long long));
		set_membership_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(set_membership, num_query_indices, query_adjacent_candidates, minidtw_size, max_warp_proportion);

		short* return_membership = (short*)malloc(sizeof(short)*100);
		cudaMemcpy(return_membership, set_membership, sizeof(short)*100, cudaMemcpyDeviceToHost);		CUERR("Copying membership from device to host")

		REQUIRE( return_membership[0] == 0 );
		REQUIRE( return_membership[1] == -1 );

		// std::cerr << "Memberships: ";
		// for(int i = 0; i < 100; i++){
			// std::cerr << return_membership[i] << ", ";
		// }
		// std::cerr << std::endl;
		
		cudaFree(set_membership);		CUERR("Free cude membership")
		cudaFree(query_adjacent_candidates);		CUERR("Free cude candidates")
		cudaFree(num_members);	CUERR("Free num members");

		free(QAC_host);
		free(return_membership);
		std::cerr << std::endl;
	}

}

//void void get_sorted_colinear_distances_kernel(QTYPE_ACC *sorted_colinear_distances, T* membership, QTYPE_ACC* query_adjacent_distances, int num_candidate_query_indices);
TEST_CASE("Sorted Colinear Distances"){

	long subject_length = 100;
	QTYPE *subject = (QTYPE*)malloc(sizeof(QTYPE)*subject_length);

	for(int i = 0; i < subject_length; i++){
		subject[i] = i;
	}

	load_subject(subject, 0, subject_length, false);

	SECTION("Good Data"){

		std::cerr << "------TEST SORTED COLINEAR DISTANCES GOOD DATA------" << std::endl;

		int dtw_dist_size = 900;
		int query_size = 10;
		int num_candidate_query_indices = 10;

		int threadblock_size = int(query_size);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual

		// short* set_mem = (short*) malloc(sizeof(short)*query_size);
		short set_mem[] = {0, -1, 0, -1, 0, -1, 0, -1, 0, -1};
		QTYPE_ACC *dtw_dists = (QTYPE_ACC*)malloc(sizeof(QTYPE_ACC)*dtw_dist_size);

		int counter = 0;

		for(int i = 0; i < dtw_dist_size; i++){
			dtw_dists[i] = subject_length - counter % subject_length;
			counter++;
			if(counter == 90) counter = 0;
		}
		for(int i = 0; i < query_size; i++){
			if(i % 2 == 0){
				set_mem[i] = 0;
			} else{
				set_mem[i] = -1;
			}
		}

		// std::cerr << "Distances: ";
		// for(int i = 0; i < dtw_dist_size; i++){
			// std::cerr << dtw_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		QTYPE_ACC *sorted_colinear_distances;
		cudaMalloc(&sorted_colinear_distances, sizeof(QTYPE_ACC)*num_candidate_query_indices);			CUERR("Allocating sorted dists");

		short* membership;
		cudaMalloc(&membership, sizeof(short)*query_size);			CUERR("Allocating membership");
		cudaMemcpy(membership, set_mem, sizeof(short)*query_size, cudaMemcpyHostToDevice); 		CUERR("Copying membership from host to device");

		QTYPE_ACC *query_adjacent_distances;
		cudaMalloc(&query_adjacent_distances, sizeof(QTYPE_ACC)*dtw_dist_size);			CUERR("Allocating dists");
		cudaMemcpy(query_adjacent_distances, dtw_dists, sizeof(QTYPE_ACC)*dtw_dist_size, cudaMemcpyHostToDevice); 		CUERR("Copying dists from host to device");

		unsigned int* total_num_members = 0;
		cudaMalloc((void**)&total_num_members, sizeof(unsigned int));		CUERR("Allocate num true members");

		int* num_sorted_colinear_distances = 0;
		cudaMalloc((void**)&num_sorted_colinear_distances, sizeof(int));		CUERR("Allocate num sorted colinear dists");

		size_t required_threadblock_shared_memory = dtw_dist_size*(sizeof(QTYPE_ACC)*2);

		get_sorted_colinear_distances_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(sorted_colinear_distances, membership, query_adjacent_distances, num_candidate_query_indices, num_sorted_colinear_distances);

		QTYPE_ACC *return_sorted_dists = (QTYPE_ACC *)malloc(sizeof(QTYPE_ACC)*num_candidate_query_indices);
		cudaMemcpy(return_sorted_dists, sorted_colinear_distances, sizeof(QTYPE_ACC)*num_candidate_query_indices, cudaMemcpyDeviceToHost); 		CUERR("Copying sorted dists from device to host");

		int return_num_sorted_dists = 0;
		cudaMemcpy(&return_num_sorted_dists, num_sorted_colinear_distances, sizeof(int), cudaMemcpyDeviceToHost); 		CUERR("Copying num of sorted dists from device to host");

		// std::cerr << "Return sorted distances: ";
		// for(int i = 0; i < num_candidate_query_indices; i++){
			// std::cerr << return_sorted_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE(return_sorted_dists[0] == 92);
		REQUIRE(return_sorted_dists[4] == 100);

		REQUIRE(return_num_sorted_dists == 5);

		cudaFree(sorted_colinear_distances);        CUERR("Freeing sorted dists");
		cudaFree(membership);                   CUERR("Freeing membership");
		cudaFree(query_adjacent_distances);                    CUERR("Freeing dists");
		cudaFree(total_num_members);        CUERR("Freeing cands");

		// free(set_mem);
		// free(dtw_dists);
		// free(query_adjacent_cands);
		free(return_sorted_dists);

		std::cerr << std::endl;

	}

	free(subject);

}

//void get_sorted_non_colinear_distances_kernel(QTYPE_ACC *sorted_non_colinear_distances, T* membership, const QTYPE_ACC *query_adjacent_distances, int* total_num_sorted_non_colinear_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, const int expected_num_pvals, int* non_colinear_distance_lengths, bool thorough_calc)
TEST_CASE("Sorted NonColinear Distances"){

	long subject_length = CUDA_THREADBLOCK_MAX_THREADS * 5;

	int dtw_dist_size = subject_length;

	int num_candidate_query_indices = CUDA_THREADBLOCK_MAX_THREADS;
	int num_candidate_subject_indices = DIV_ROUNDUP(subject_length/MINIDTW_STRIDE, CUDA_THREADBLOCK_MAX_THREADS);

	bool thorough_calc = true;
	short* set_mem = (short*) malloc(sizeof(short)*num_candidate_query_indices);
	QTYPE_ACC *dtw_dists = (QTYPE_ACC*)malloc(sizeof(QTYPE_ACC)*dtw_dist_size);

	int counter = 0;
	for(int i = 0; i < dtw_dist_size; i++){
		dtw_dists[i] = counter + (i / 256);
		if(++counter == 256) counter = 0;
	}
	for(int i = 0; i < num_candidate_query_indices; i++){
		set_mem[i] = i % 2 == 0 ? 0 : -1;
	}		
		
	// std::cerr << "Distances: ";
	// for(int i = 0; i < dtw_dist_size; i++){
		// std::cerr << dtw_dists[i] << ", ";
	// }
	// std::cerr << std::endl;

	short* membership;
	cudaMalloc(&membership, sizeof(short)*num_candidate_query_indices);			CUERR("Allocating membership");
	cudaMemcpy(membership, set_mem, sizeof(short)*num_candidate_query_indices, cudaMemcpyHostToDevice); 		CUERR("Copying membership from host to device");

	QTYPE_ACC *query_adjacent_distances;
	cudaMalloc(&query_adjacent_distances, sizeof(QTYPE_ACC)*dtw_dist_size);			CUERR("Allocating dists");
	cudaMemcpy(query_adjacent_distances, dtw_dists, sizeof(QTYPE_ACC)*dtw_dist_size, cudaMemcpyHostToDevice); 		CUERR("Copying dists from host to device");

	int* total_num_sorted_non_colinear_distances;
	int num_non_dists = 0;
	cudaMalloc(&total_num_sorted_non_colinear_distances, sizeof(int));	CUERR("Allocating total_num_sorted_non_colinear_distances");
	cudaMemcpy(total_num_sorted_non_colinear_distances, &num_non_dists, sizeof(int), cudaMemcpyHostToDevice); 		CUERR("Copying total_num_sorted_non_colinear_distances from host to device");

	SECTION("Good Data With Equal Colinear and Non Colinear Distances"){

		std::cerr << "------TEST SORTED NON COLINEAR DISTANCES GOOD DATA EQUAL COLINEAR/ NONCOLINEAR------" << std::endl;

		int num_sorted_colinear_distances = num_candidate_query_indices;

		QTYPE_ACC *sorted_non_colinear_distances;
		int num_sorted_non_colinear_distances = 2*num_candidate_query_indices - num_sorted_colinear_distances;
		cudaMalloc(&sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);			CUERR("Allocating sorted non col dists");

		int threadblock_size = int(num_candidate_query_indices);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual
		size_t required_threadblock_shared_memory = dtw_dist_size*sizeof(QTYPE_ACC);

		int *non_colinear_distance_lengths;
		cudaMalloc(&non_colinear_distance_lengths, sizeof(int)*pval_griddim.x);	CUERR("Allocating non_colinear_distance_lengths");

		get_sorted_non_colinear_distances_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(sorted_non_colinear_distances, membership, query_adjacent_distances, total_num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, non_colinear_distance_lengths, num_sorted_colinear_distances, thorough_calc);

		QTYPE_ACC *return_sorted_dists = (QTYPE_ACC *)malloc(sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);
		cudaMemcpy(return_sorted_dists, sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances, cudaMemcpyDeviceToHost); 		CUERR("Copying sorted dists from device to host");

		// std::cerr << "Return sorted non distances: ";
		// for(int i = 0; i < num_sorted_non_colinear_distances; i++){
			// std::cerr << return_sorted_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE(return_sorted_dists[0] == 1);
		REQUIRE(return_sorted_dists[3] == 4);

		cudaFree(sorted_non_colinear_distances);        CUERR("Freeing sorted dists");

		// free(query_adjacent_cands);
		free(return_sorted_dists);

		std::cerr << std::endl;

	}


	SECTION("Good Data With More Colinear and Less Non Colinear Distances"){

		std::cerr << "------TEST SORTED NON COLINEAR DISTANCES GOOD DATA MORE COLINEAR/ LESS NONCOLINEAR------" << std::endl;

		int num_sorted_colinear_distances = 300;

		QTYPE_ACC *sorted_non_colinear_distances;
		int num_sorted_non_colinear_distances = 2*num_candidate_query_indices - num_sorted_colinear_distances;
		cudaMalloc(&sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);			CUERR("Allocating sorted non col dists");

		int threadblock_size = int(num_candidate_query_indices);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual
		size_t required_threadblock_shared_memory = dtw_dist_size*sizeof(QTYPE_ACC);

		int *non_colinear_distance_lengths;
		cudaMalloc(&non_colinear_distance_lengths, sizeof(int)*pval_griddim.x);	CUERR("Allocating non_colinear_distance_lengths");

		get_sorted_non_colinear_distances_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(sorted_non_colinear_distances, membership, query_adjacent_distances, total_num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, non_colinear_distance_lengths, num_sorted_colinear_distances, thorough_calc);

		QTYPE_ACC *return_sorted_dists = (QTYPE_ACC *)malloc(sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);
		cudaMemcpy(return_sorted_dists, sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances, cudaMemcpyDeviceToHost); 		CUERR("Copying sorted dists from device to host");

		// std::cerr << "Return sorted non distances: ";
		// for(int i = 0; i < num_sorted_non_colinear_distances; i++){
			// std::cerr << return_sorted_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE(return_sorted_dists[0] == 1);
		REQUIRE(return_sorted_dists[1] == 2);
		REQUIRE(return_sorted_dists[2] == 3);
		REQUIRE(return_sorted_dists[166] == 167);
		REQUIRE(return_sorted_dists[167] == 168);
		REQUIRE(return_sorted_dists[168] == 169);
		REQUIRE(return_sorted_dists[169] == 171);

		cudaFree(sorted_non_colinear_distances);        CUERR("Freeing sorted dists");

		// free(query_adjacent_cands);
		free(return_sorted_dists);

		std::cerr << std::endl;

	}

	SECTION("Good Data With Less Colinear and More Non Colinear Distances"){

		std::cerr << "------TEST SORTED NON COLINEAR DISTANCES GOOD DATA LESS COLINEAR/ MORE NONCOLINEAR------" << std::endl;

		int num_sorted_colinear_distances = 212;

		QTYPE_ACC *sorted_non_colinear_distances;
		int num_sorted_non_colinear_distances = 2*num_candidate_query_indices - num_sorted_colinear_distances;
		cudaMalloc(&sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);			CUERR("Allocating sorted non col dists");

		int threadblock_size = int(num_candidate_query_indices);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual
		size_t required_threadblock_shared_memory = dtw_dist_size*sizeof(QTYPE_ACC);

		int *non_colinear_distance_lengths;
		cudaMalloc(&non_colinear_distance_lengths, sizeof(int)*pval_griddim.x);	CUERR("Allocating non_colinear_distance_lengths");

		get_sorted_non_colinear_distances_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(sorted_non_colinear_distances, membership, query_adjacent_distances, total_num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, non_colinear_distance_lengths, num_sorted_colinear_distances, thorough_calc);

		QTYPE_ACC *return_sorted_dists = (QTYPE_ACC *)malloc(sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);
		cudaMemcpy(return_sorted_dists, sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances, cudaMemcpyDeviceToHost); 		CUERR("Copying sorted dists from device to host");

		// std::cerr << "Return sorted non distances: ";
		// for(int i = 0; i < num_sorted_non_colinear_distances; i++){
			// std::cerr << return_sorted_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE(return_sorted_dists[0] == 1);
		REQUIRE(return_sorted_dists[1] == 2);
		REQUIRE(return_sorted_dists[2] == 3);
		REQUIRE(return_sorted_dists[3] == 3);
		REQUIRE(return_sorted_dists[10] == 9);
		REQUIRE(return_sorted_dists[11] == 10);
		REQUIRE(return_sorted_dists[12] == 11);
		REQUIRE(return_sorted_dists[13] == 11);

		cudaFree(sorted_non_colinear_distances);        CUERR("Freeing sorted dists");

		// free(query_adjacent_cands);
		free(return_sorted_dists);

		std::cerr << std::endl;

	}

	SECTION("Good Data With Less Colinear and More Non Colinear Distances"){

		std::cerr << "------TEST SORTED NON COLINEAR DISTANCES TWO MEMBERSHIPS------" << std::endl;

		short* set_mem_test = (short*) malloc(sizeof(short)*num_candidate_query_indices);
	
		for(int i = 0; i < num_candidate_query_indices; i++){
			set_mem_test[i] = i % 2 == 0 ? 0 : 1;
		}		
	
		short* membership_test;
		cudaMalloc(&membership_test, sizeof(short)*num_candidate_query_indices);			CUERR("Allocating membership");
		cudaMemcpy(membership_test, set_mem_test, sizeof(short)*num_candidate_query_indices, cudaMemcpyHostToDevice); 		CUERR("Copying membership from host to device");


		int num_sorted_colinear_distances = 256;

		QTYPE_ACC *sorted_non_colinear_distances;
		int num_sorted_non_colinear_distances = 2*num_candidate_query_indices - num_sorted_colinear_distances;
		cudaMalloc(&sorted_non_colinear_distances, sizeof(QTYPE_ACC)*2*num_sorted_non_colinear_distances);			CUERR("Allocating sorted non col dists");

		int threadblock_size = int(num_candidate_query_indices);
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual
		size_t required_threadblock_shared_memory = dtw_dist_size*sizeof(QTYPE_ACC);

		int *non_colinear_distance_lengths;
		cudaMalloc(&non_colinear_distance_lengths, sizeof(int)*pval_griddim.x);	CUERR("Allocating non_colinear_distance_lengths");

		get_sorted_non_colinear_distances_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(sorted_non_colinear_distances, membership_test, query_adjacent_distances, total_num_sorted_non_colinear_distances, num_candidate_query_indices, num_candidate_subject_indices, non_colinear_distance_lengths, num_sorted_colinear_distances, thorough_calc);

		QTYPE_ACC *return_sorted_dists = (QTYPE_ACC *)malloc(sizeof(QTYPE_ACC)*2*num_sorted_non_colinear_distances);
		cudaMemcpy(return_sorted_dists, sorted_non_colinear_distances, sizeof(QTYPE_ACC)*2*num_sorted_non_colinear_distances, cudaMemcpyDeviceToHost); 		CUERR("Copying sorted dists from device to host");

		// std::cerr << "Return sorted non distances: ";
		// for(int i = 0; i < 2*num_sorted_non_colinear_distances; i++){
			// std::cerr << return_sorted_dists[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE(return_sorted_dists[0] == 1);
		REQUIRE(return_sorted_dists[1] == 2);
		REQUIRE(return_sorted_dists[2] == 3);
		REQUIRE(return_sorted_dists[num_sorted_non_colinear_distances] == 2);
		REQUIRE(return_sorted_dists[num_sorted_non_colinear_distances+1] == 3);
		REQUIRE(return_sorted_dists[num_sorted_non_colinear_distances+2] == 4);

		cudaFree(sorted_non_colinear_distances);        CUERR("Freeing sorted dists");
		cudaFree(membership_test);        CUERR("Freeing membership test");

		// free(query_adjacent_cands);
		free(return_sorted_dists);
		free(set_mem_test);

		std::cerr << std::endl;

	}

	
	free(set_mem);
	free(dtw_dists);

	cudaFree(membership);                   CUERR("Freeing membership");
	cudaFree(query_adjacent_distances);                    CUERR("Freeing dists");

	// free(subject);

}

// void calculate_pval_kernel(const float max_pval, int query_adjacent_candidates, short set_membership, QTYPE_ACC *sorted_colinear_distances, QTYPE_ACC *sorted_non_colinear_distances, int num_sorted_non_colinear_distances, unsigned int *num_results_recorded, unsigned int *num_results_notrecorded, unsigned int max_num_results, float *output_pvals, int *output_left_anchors_query, int *output_right_anchors_query, int *output_left_anchors_subject, int *output_right_anchors_subject, int *output_num_members, int first_idx);
TEST_CASE(" Calculate Pvals "){

	SECTION(" Good Data "){

		std::cerr << "------TEST CALC PVALS GOOD DATA------" << std::endl;

		int num_sorted_colinear_distances = 10;
		int num_sorted_non_colinear_distances = 502;

		int minidtw_size = 10;

		QTYPE_ACC *sorted_colinear_dists = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*num_sorted_colinear_distances);
		QTYPE_ACC *sorted_non_colinear_dists = (QTYPE_ACC*) malloc(sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);
		
		std::string good_colinear_file = current_working_dir + "/good_files/text/random_distances_colinear.txt";
		std::string good_noncolinear_file = current_working_dir + "/good_files/text/random_distances_non_colinear.txt";

		std::string line;
		std::ifstream colinear_file(good_colinear_file);
		int counter = 0;
		// std::cerr << "Colinear: ";
		if (colinear_file.is_open()){
			while (getline(colinear_file,line) && counter < num_sorted_colinear_distances){
				sorted_colinear_dists[counter] = std::atof(line.c_str());
				// std::cerr << sorted_colinear_dists[counter] << ", ";
				counter++;
			}
			colinear_file.close();
		}
		// std::cerr << std::endl;
		std::ifstream noncolinear_file(good_noncolinear_file);
		counter = 0;
		// std::cerr << "Noncolinear: ";
		if (noncolinear_file.is_open()){
			while (getline(noncolinear_file,line) && counter < num_sorted_non_colinear_distances){
				sorted_non_colinear_dists[counter] = std::atof(line.c_str());
				// std::cerr << sorted_non_colinear_dists[counter] << ", ";
				counter++;
			}
			noncolinear_file.close();
		}
		// std::cerr << std::endl;

		QTYPE_ACC *sorted_colinear_distances;
		cudaMalloc(&sorted_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_colinear_distances);														CUERR("Malloc sorted colinear distances");	
		cudaMemcpy(sorted_colinear_distances, sorted_colinear_dists, sizeof(QTYPE_ACC)*num_sorted_colinear_distances, cudaMemcpyHostToDevice);		CUERR("Copy colinear distances from host to device");

		QTYPE_ACC *sorted_non_colinear_distances;
		cudaMalloc(&sorted_non_colinear_distances, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances);														CUERR("Malloc sorted non colinear distances");	
		cudaMemcpy(sorted_non_colinear_distances, sorted_non_colinear_dists, sizeof(QTYPE_ACC)*num_sorted_non_colinear_distances, cudaMemcpyHostToDevice);	CUERR("Copy non colinear distances from host to device");

		std::vector<int> indices{0, 3, 6, 8, 9, 10, 16, 18, 31, 38};

		int query_size = 512;
		int* query_cands = (int*) malloc(sizeof(int)*query_size);
		short* set_mem = (short*) malloc(sizeof(short)*query_size);
		for(int i = 0; i < query_size; i++){
			query_cands[i] = i;
		}
		for(int i = 0; i < query_size; i++){
			if(std::find(indices.begin(), indices.end(), i) != indices.end()){
				set_mem[i] = 0;
			} else{
				set_mem[i] = -1;
			}
		}

		// std::cerr << "Memberships: ";
		// for(int i = 0; i < 3000; i++){
			// std::cerr << set_mem[i] << ", ";
		// }
		// std::cerr << std::endl;

		int* query_adjacent_candidates;
		cudaMalloc(&query_adjacent_candidates, sizeof(int)*query_size);  	CUERR("Allocate memory for candidates"); 
		cudaMemcpy(query_adjacent_candidates, query_cands, sizeof(int)*query_size, cudaMemcpyHostToDevice);		CUERR("Copying candidates fron host to device")
		short* set_membership;
		cudaMalloc(&set_membership, sizeof(short)*query_size);  	CUERR("Allocate memory for membership"); 
		cudaMemcpy(set_membership, set_mem, sizeof(short)*query_size, cudaMemcpyHostToDevice);		CUERR("Copying membership from host to device")

		float max_pval = 1.0;

		float *output_pvals;
		int *output_num_members = 0;
		int *output_left_anchors_query, *output_right_anchors_query;
		long *output_left_anchors_subject, *output_right_anchors_subject;
		unsigned int *num_results_recorded = 0;
		unsigned int *num_results_notrecorded = 0;
		cudaMalloc(&num_results_recorded, sizeof(unsigned int));                                   CUERR("Allocating GPU memory for DTW anchor kept count");
		cudaMalloc(&num_results_notrecorded, sizeof(unsigned int));                              CUERR("Allocating GPU memory for DTW anchor discarded count");
		cudaMalloc(&output_pvals, sizeof(float)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for DTW anchor p-values");
		cudaMalloc(&output_num_members, sizeof(int)*MAX_PVAL_MATCHES_KEPT);              CUERR("Allocating GPU memory for set membership");
		cudaMalloc(&output_left_anchors_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' first DTW anchors");
		cudaMalloc(&output_right_anchors_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' last DTW anchors");
		cudaMalloc(&output_left_anchors_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' first DTW anchors");
		cudaMalloc(&output_right_anchors_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for sets' last DTW anchors");

		int* num_sorted_non_colinear_distances_device;
		cudaMalloc(&num_sorted_non_colinear_distances_device, sizeof(int));                                   CUERR("Allocating GPU memory for num_sorted_non_colinear_distances_device");
		cudaMemcpy(num_sorted_non_colinear_distances_device, &num_sorted_non_colinear_distances, sizeof(int), cudaMemcpyHostToDevice);		CUERR("Copying num_sorted_non_colinear_distances from host to device")

		int threadblock_size = 100;
		// int td = threadblock_size;
		dim3 pval_griddim(1, 1, 1); // really big queries may get split, also round up as per usual

		size_t required_threadblock_shared_memory = num_sorted_non_colinear_distances*sizeof(long long);

		calculate_pval_kernel<<<pval_griddim, threadblock_size, required_threadblock_shared_memory, 0>>>(
						max_pval, query_adjacent_candidates, set_membership,
						sorted_colinear_distances, sorted_non_colinear_distances, 
						num_sorted_non_colinear_distances_device, 0,
						num_results_recorded, num_results_notrecorded, MAX_PVAL_MATCHES_KEPT, 
						output_pvals, output_left_anchors_query, output_right_anchors_query, 	
						output_left_anchors_subject, output_right_anchors_subject, 
						output_num_members, true, false,
						0, 0, 0, 0, 0, 0, 10, minidtw_size);								CUERR("Running pval kernel")

		unsigned int return_num_results_recorded;
		unsigned int return_num_results_notrecorded;
		float* return_output_pvals = (float*) malloc(sizeof(float)*MAX_PVAL_MATCHES_KEPT);
		int* return_output_num_members = (int*) malloc(sizeof(int)*MAX_PVAL_MATCHES_KEPT);
		int* return_output_left_anchors_query = (int*) malloc(sizeof(int)*MAX_PVAL_MATCHES_KEPT);
		int* return_output_right_anchors_query = (int*) malloc(sizeof(int)*MAX_PVAL_MATCHES_KEPT);
		long* return_output_left_anchors_subject = (long*) malloc(sizeof(long)*MAX_PVAL_MATCHES_KEPT);
		long* return_output_right_anchors_subject = (long*) malloc(sizeof(long)*MAX_PVAL_MATCHES_KEPT);

		cudaMemcpy(&return_num_results_recorded, num_results_recorded, sizeof(unsigned int), cudaMemcpyDeviceToHost);									CUERR("Copying num results recorded from device to host")
		cudaMemcpy(&return_num_results_notrecorded, num_results_notrecorded, sizeof(unsigned int), cudaMemcpyDeviceToHost);								CUERR("Copying num results not recorded from device to host")
		cudaMemcpy(return_output_pvals, output_pvals, sizeof(float)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);										CUERR("Copying pvals from device to host")
		cudaMemcpy(return_output_num_members, output_num_members, sizeof(int)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);							CUERR("Copying members from device to host")
		cudaMemcpy(return_output_left_anchors_query, output_left_anchors_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);				CUERR("Copying left query from device to host")
		cudaMemcpy(return_output_right_anchors_query, output_right_anchors_query, sizeof(int)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);			CUERR("Copying right query from device to host")
		cudaMemcpy(return_output_left_anchors_subject, output_left_anchors_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);			CUERR("Copying left subject from device to host")
		cudaMemcpy(return_output_right_anchors_subject, output_right_anchors_subject, sizeof(long)*MAX_PVAL_MATCHES_KEPT, cudaMemcpyDeviceToHost);		CUERR("Copying right subject from device to host")

		REQUIRE( return_num_results_recorded == 1 );
		REQUIRE( return_num_results_notrecorded == 0 );
		REQUIRE( return_output_left_anchors_query[0] == 0 );
		REQUIRE( return_output_right_anchors_query[0] == 90 );
		REQUIRE( return_output_left_anchors_subject[0] == 0 );
		REQUIRE( return_output_right_anchors_subject[0] == 9 );

		std::cerr << "return_output_pvals: ";
		for(int i = 0; i < return_num_results_recorded; i++){
			std::cerr << return_output_pvals[i] << ", ";
		}
		std::cerr << std::endl;
		std::cerr << "return_output_num_members: ";
		for(int i = 0; i < return_num_results_recorded; i++){
			std::cerr << return_output_num_members[i] << ", ";
		}
		std::cerr << std::endl;

		cudaFree(num_results_recorded);              CUERR("Freeing num results recorded")
		cudaFree(num_results_notrecorded);           CUERR("Freeing num results not recorded")
		cudaFree(output_pvals);                      CUERR("Freeing pvals")
		cudaFree(output_num_members);                CUERR("Freeing members")
		cudaFree(output_left_anchors_query);         CUERR("Freeing left query")
		cudaFree(output_right_anchors_query);        CUERR("Freeing right query")
		cudaFree(output_left_anchors_subject);	     CUERR("Freeing left subject")
		cudaFree(output_right_anchors_subject);		 CUERR("Freeing right subject")

		free(return_output_pvals);
		free(return_output_num_members);
		free(return_output_left_anchors_query);
		free(return_output_right_anchors_query);
		free(return_output_left_anchors_subject);
		free(return_output_right_anchors_subject);

		cudaFree(query_adjacent_candidates);      CUERR("Freeing query cands")
		cudaFree(set_membership);                 CUERR("Freeing members")

		free(query_cands);
		free(set_mem);

		free(sorted_colinear_dists);
		free(sorted_non_colinear_dists);

		cudaFree(sorted_colinear_distances);		 CUERR("Freeing colinear distances");
		cudaFree(sorted_non_colinear_distances);     CUERR("Freeing non colinear distances");

		std::cerr << std::endl;

	}

}

TEST_CASE(" Welford Znorm "){
	SECTION("Good Data"){
		std::cerr << "------TEST WELFORD ZNORM GOOD DATA------" << std::endl;

		float* welford_mean = (float*) malloc(sizeof(float));
		float* welford_ssq = (float*) malloc(sizeof(float));
		(*welford_mean) = 0;
		(*welford_ssq) = 0;

		float* tmp_mean, *tmp_ssq;

		int query_length = 10;
		QTYPE query[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 
						  0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 
						  0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 
						  0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 
						  0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 
						  0, 0, 0, 0, 100, 100, 100, 100, 100, 100,
						  0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 
						  0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 
						  0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
						  100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };

		cudaMalloc(&tmp_mean, sizeof(float)); CUERR("Allocating GPU memory for temp welford mean");
		cudaMalloc(&tmp_ssq, sizeof(float)); CUERR("Allocating GPU memory for temp welford ssq");

		long total_values_znormalized = 0;

		for(int i = 0; i < 100; i+=10){
			
			load_G_query<QTYPE>(&query[i], query_length);
			cudaMemcpy(tmp_mean, welford_mean, sizeof(float), cudaMemcpyHostToDevice);	CUERR("Copying welford mean from host to device");
			cudaMemcpy(tmp_ssq, welford_ssq, sizeof(float), cudaMemcpyHostToDevice);		CUERR("Copying welford ssq from host to device");
			welford_query_znorm<<<1, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(0, query_length, tmp_mean, tmp_ssq, total_values_znormalized);
			total_values_znormalized += query_length;
			cudaMemcpy(welford_mean, tmp_mean, sizeof(float), cudaMemcpyDeviceToHost);	CUERR("Copying welford mean from device to host");
			cudaMemcpy(welford_ssq, tmp_ssq, sizeof(float), cudaMemcpyDeviceToHost);		CUERR("Copying welford ssq from device to host");
			
		}
		REQUIRE( ceil((*welford_mean)) == 55 ); // Should be 55 but this is what we get so close enough. Seems like REQUIRE acts weird for floats
		REQUIRE( ceil((*welford_ssq)) == 247500 );	// Same problem as mentioned above
		

		cudaFree(tmp_mean);	CUERR("Freeing temp welford mean");
		cudaFree(tmp_ssq);	CUERR("Freeing temp welford ssq");
		free(welford_mean);
		free(welford_ssq);

		std::cerr << std::endl;
	}
}

// void variance(T *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances);
TEST_CASE(" Variance "){

	SECTION("Good Data"){
		std::cerr << "------TEST VARIANCE GOOD DATA------" << std::endl;

		long data_length = 10;
		float CPU_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	
		float* GPU_data;
		cudaMalloc(&GPU_data, sizeof(float)*data_length);
		cudaMemcpy(GPU_data, CPU_data, sizeof(float)*data_length, cudaMemcpyHostToDevice);	CUERR("Copying data from CPU to GPU");

		int num_threadblocks = DIV_ROUNDUP(data_length, CUDA_THREADBLOCK_MAX_THREADS);

		float data_mean = 5.5;
		float *threadblock_means;
		cudaMalloc(&threadblock_means, sizeof(float)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);               CUERR("Allocating device memory for query Z-norm threadblock means");
		cudaMemcpy(threadblock_means, &data_mean, sizeof(float), cudaMemcpyHostToDevice);		CUERR("Copying mean frome host to device");

		float *threadblock_results;
		// There is a chance that these variances sums could get really big for very long datasets, TODO we may want to store into a double rather than float if QTYPE = float
		cudaMalloc(&threadblock_results, sizeof(float)*num_threadblocks);                CUERR("Allocating device memory for query Z-norm threadblock variances");

		dim3 grid(num_threadblocks, 1, 1);
		int req_threadblock_shared_memory = sizeof(float)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
		variance<float><<<grid, CUDA_THREADBLOCK_MAX_THREADS, req_threadblock_shared_memory, 0>>>(GPU_data, data_length, data_length, threadblock_means, threadblock_results);

		float* return_results = (float*)malloc(sizeof(float)*num_threadblocks);
		cudaMemcpy(return_results, threadblock_results, sizeof(float)*num_threadblocks, cudaMemcpyDeviceToHost);	CUERR("Copying results from GPU to CPU");

		REQUIRE( round_to_three(return_results[0]) == 2.872f );

		cudaFree(GPU_data);		CUERR("Freeing data");
		cudaFree(threadblock_means);		CUERR("Freeing means");
		cudaFree(threadblock_results);		CUERR("Freeing results");

		std::cerr << std::endl;
	}

	SECTION("More Data"){
		std::cerr << "------TEST VARIANCE MORE DATA------" << std::endl;

		long data_length = CUDA_THREADBLOCK_MAX_THREADS*2;
		float* CPU_data = (float*)malloc(sizeof(float)*data_length);
		for(int i = 0; i < data_length; i++){
			CPU_data[i] = i+1;
		}
	
		float* GPU_data;
		cudaMalloc(&GPU_data, sizeof(float)*data_length);
		cudaMemcpy(GPU_data, CPU_data, sizeof(float)*data_length, cudaMemcpyHostToDevice);	CUERR("Copying data from CPU to GPU");

		int num_threadblocks = DIV_ROUNDUP(data_length, CUDA_THREADBLOCK_MAX_THREADS);

		float data_mean = 256.5;
		float *threadblock_means;
		cudaMalloc(&threadblock_means, sizeof(float)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);               CUERR("Allocating device memory for query Z-norm threadblock means");
		cudaMemcpy(threadblock_means, &data_mean, sizeof(float), cudaMemcpyHostToDevice);		CUERR("Copying mean frome host to device");

		float *threadblock_results;
		// There is a chance that these variances sums could get really big for very long datasets, TODO we may want to store into a double rather than float if QTYPE = float
		cudaMalloc(&threadblock_results, sizeof(float)*num_threadblocks);                CUERR("Allocating device memory for query Z-norm threadblock variances");

		dim3 grid(num_threadblocks, 1, 1);
		int req_threadblock_shared_memory = sizeof(float)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
		variance<float><<<grid, CUDA_THREADBLOCK_MAX_THREADS, req_threadblock_shared_memory, 0>>>(GPU_data, data_length, data_length, threadblock_means, threadblock_results);

		float* return_results = (float*)malloc(sizeof(float)*num_threadblocks);
		cudaMemcpy(return_results, threadblock_results, sizeof(float)*num_threadblocks, cudaMemcpyDeviceToHost);	CUERR("Copying results from GPU to CPU");

		REQUIRE( return_results[0] == 5592384);
		REQUIRE( return_results[1] == 5592384);


		cudaFree(GPU_data);		CUERR("Freeing data");
		cudaFree(threadblock_means);		CUERR("Freeing means");
		cudaFree(threadblock_results);		CUERR("Freeing results");

		std::cerr << std::endl;
	}
}

// void variance_float(float *data, long data_length, long orig_data_length, float *data_mean, float *threadblock_variances);
TEST_CASE(" Variance Float "){

	SECTION("Good Data"){
		std::cerr << "------TEST VARIANCE FLOAT GOOD DATA------" << std::endl;

		long original_data_length = CUDA_THREADBLOCK_MAX_THREADS*2;

		long current_data_length = 2;
		float CPU_data[] = { 5592384, 5592384 };
	
		float* GPU_data;
		cudaMalloc(&GPU_data, sizeof(float)*current_data_length);
		cudaMemcpy(GPU_data, CPU_data, sizeof(float)*current_data_length, cudaMemcpyHostToDevice);	CUERR("Copying data from CPU to GPU");

		int num_threadblocks = DIV_ROUNDUP(original_data_length, CUDA_THREADBLOCK_MAX_THREADS);

		float data_mean = 5.5;
		float *threadblock_means;
		cudaMalloc(&threadblock_means, sizeof(float)*num_threadblocks*CUDA_THREADBLOCK_MAX_THREADS);               CUERR("Allocating device memory for query Z-norm threadblock means");
		cudaMemcpy(threadblock_means, &data_mean, sizeof(float), cudaMemcpyHostToDevice);		CUERR("Copying mean frome host to device");

		dim3 grid(DIV_ROUNDUP(num_threadblocks,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		int req_threadblock_shared_memory = sizeof(float)*CUDA_THREADBLOCK_MAX_THREADS/CUDA_WARP_WIDTH;
		int threads = num_threadblocks > CUDA_THREADBLOCK_MAX_THREADS ? CUDA_THREADBLOCK_MAX_THREADS : num_threadblocks;
		variance_float<<<grid, threads, req_threadblock_shared_memory, 0>>>(GPU_data, current_data_length, original_data_length, threadblock_means, GPU_data);

		float* return_results = (float*)malloc(sizeof(float)*num_threadblocks);
		cudaMemcpy(return_results, GPU_data, sizeof(float)*num_threadblocks, cudaMemcpyDeviceToHost);	CUERR("Copying results from GPU to CPU");

		REQUIRE( round_to_three(return_results[0]) == 147.801f );
		REQUIRE( return_results[1] == 5592384);

		cudaFree(GPU_data);		CUERR("Freeing data");
		cudaFree(threadblock_means);		CUERR("Freeing means");

		std::cerr << std::endl;
	}
}

// void load_non_colinear_distances(const QTYPE_ACC *query_adjacent_distances, const int num_candidate_query_indices, const int num_candidate_subject_indices, QTYPE_ACC *global_non_colinear_distances, int *num_non_colinear_distances);
TEST_CASE(" Load Non Colinear Distances "){

	SECTION("Good Data"){
		std::cerr << "------TEST LOAD NON COLINEAR DISTANCES GOOD DATA------" << std::endl;

		long subject_length = CUDA_THREADBLOCK_MAX_THREADS * 5;

		int dtw_dist_size = subject_length;
		
		QTYPE_ACC *dtw_dists = (QTYPE_ACC*)malloc(sizeof(QTYPE_ACC)*dtw_dist_size);
		
		int counter = 0;
		for(int i = 0; i < dtw_dist_size; i++){
			dtw_dists[i] = counter + (i / 256);
			if(++counter == 256) counter = 0;
		}
		
		QTYPE_ACC *query_adjacent_distances;
		cudaMalloc(&query_adjacent_distances, sizeof(QTYPE_ACC)*dtw_dist_size);			CUERR("Allocating dists");
		cudaMemcpy(query_adjacent_distances, dtw_dists, sizeof(QTYPE_ACC)*dtw_dist_size, cudaMemcpyHostToDevice); 		CUERR("Copying dists from host to device");
		
		long num_query_indices = CUDA_THREADBLOCK_MAX_THREADS;

		int num_subject_buckets_per_query_minidtw = DIV_ROUNDUP(subject_length/MINIDTW_STRIDE, CUDA_THREADBLOCK_MAX_THREADS);
		QTYPE_ACC *global_non_colinear_distances;
		int *num_non_colinear_distances;

		cudaMalloc(&global_non_colinear_distances, sizeof(QTYPE_ACC)*1024);
		cudaMalloc(&num_non_colinear_distances, sizeof(int));
		
		load_non_colinear_distances<<<1, num_query_indices, 0, 0>>>(query_adjacent_distances, num_query_indices, num_subject_buckets_per_query_minidtw, global_non_colinear_distances, num_non_colinear_distances);

		QTYPE_ACC* return_distances = (QTYPE_ACC*)malloc(sizeof(QTYPE_ACC)*num_query_indices);
		int* return_num_dists = (int*)malloc(sizeof(int));

		cudaMemcpy(return_distances, global_non_colinear_distances, sizeof(QTYPE_ACC)*num_query_indices, cudaMemcpyDeviceToHost);	CUERR("Coping dists from device to host");
		cudaMemcpy(return_num_dists, num_non_colinear_distances, sizeof(int), cudaMemcpyDeviceToHost);	CUERR("Copying num dists from device to host");

		REQUIRE( *return_num_dists == 256 );
		for(int i = 0; i < num_query_indices; i++){
			REQUIRE( return_distances[i] == i );
		}

		free(dtw_dists);
		free(return_distances);
		cudaFree(query_adjacent_distances);	CUERR("Free distances");

		std::cerr << std::endl;
	}

}

// void fdr(float *input_pvals, int *input_pval_ranks, unsigned int input_pvals_length, float *output_qvals);
TEST_CASE(" FDR "){
	SECTION("Good Data"){
		std::cerr << "------TEST FDR GOOD DATA------" << std::endl;

		unsigned int num_pvals_host = 100;
		float* pvals = (float*)malloc(sizeof(float)*num_pvals_host);
		for(int i = 0; i < num_pvals_host; i++){
			pvals[i] = (float)i / (float)num_pvals_host;
		}
		int threadblock_size = num_pvals_host > CUDA_THREADBLOCK_MAX_THREADS ? CUDA_THREADBLOCK_MAX_THREADS : num_pvals_host;
		dim3 fdr_griddim(DIV_ROUNDUP(num_pvals_host, CUDA_THREADBLOCK_MAX_THREADS), 1, 1);

		float *anchor_pvals;
		float *qvals = 0;
		int *anchor_pval_ranks = 0;
		float *anchor_pvals_host = 0;
		int *anchor_pval_ranks_host = 0;

		cudaMalloc(&anchor_pvals, sizeof(float)*MAX_PVAL_MATCHES_KEPT);                             CUERR("Allocating GPU memory for DTW anchor p-values");
		cudaMalloc(&qvals, sizeof(float)*num_pvals_host); 	CUERR("Allocating GPU memory for min DTW FDRs");
		cudaMalloc(&anchor_pval_ranks, sizeof(int)*num_pvals_host);			CUERR("Allocating GPU memory for DTW anchor p-value ranks");
		cudaMallocHost(&anchor_pvals_host, num_pvals_host*sizeof(float));                             CUERR("Allocating CPU memory for DTW anchor p-values");
		cudaMallocHost(&anchor_pval_ranks_host, sizeof(int)*num_pvals_host);											 CUERR("Allocating CPU memory for DTW anchor p-value ranks");

		cudaMemcpy(anchor_pvals, pvals, sizeof(float)*num_pvals_host, cudaMemcpyHostToDevice);	CUERR("Copying pvals from device to host");

		cudaMemcpyAsync(anchor_pvals_host, anchor_pvals, sizeof(float)*num_pvals_host, cudaMemcpyDeviceToHost, 0);  CUERR("Copying DTW anchor p-values from GPU to CPU");

		cudaStreamSynchronize(0);

		int anchor_num_ranks = 1;
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
			// std::cerr << anchor_pval_ranks_host[i] << ", ";
		}
		// std::cerr << std::endl;

		cudaMemcpy(anchor_pval_ranks, anchor_pval_ranks_host, sizeof(int)*num_pvals_host, cudaMemcpyHostToDevice);			CUERR("Copying anchor ranks back to GPU")

		fdr<<<fdr_griddim, threadblock_size, 0, 0>>>(anchor_pvals, anchor_pval_ranks, num_pvals_host, qvals);    CUERR("Calculating false discovery rates");

		float* return_qvals = (float*)malloc(sizeof(float)*num_pvals_host);
		cudaMemcpy(return_qvals, qvals, sizeof(float)*num_pvals_host, cudaMemcpyDeviceToHost);	CUERR("Copying qvals from device to host");

		REQUIRE( return_qvals[0] == 0 );
		REQUIRE( return_qvals[1] == 0.5 );
		REQUIRE( round_to_three(return_qvals[98]) == 0.990f );
		REQUIRE( return_qvals[99] == 0.99f );

		free(pvals);
		free(return_qvals);

		cudaFree(anchor_pvals);				CUERR("Free pvals");
		cudaFree(qvals);                    CUERR("Free qvals");
		cudaFree(anchor_pval_ranks);        CUERR("Free ranks");
		cudaFreeHost(anchor_pvals_host);        CUERR("Free host pvals");
		cudaFreeHost(anchor_pval_ranks_host);   CUERR("Free host ranks");

		std::cerr << std::endl;
	}
}

// float warpReduceSum(float val);
TEST_CASE(" Warp Reduce Sum "){
	SECTION("Good Data"){
		std::cerr << "------TEST WARP REDUCE SUM GOOD DATA------" << std::endl;

		int num_values = 32;
		float* host_vals = (float*)malloc(sizeof(float)*num_values);
		for(int i = 0; i < num_values; i++){
			host_vals[i] = i+1;
		}

		float* device_vals;
		cudaMalloc(&device_vals, sizeof(float)*num_values);		CUERR("Allocate memory for values");
		cudaMemcpy(device_vals, host_vals, sizeof(float)*num_values, cudaMemcpyHostToDevice);	CUERR("Copy values from host to device");

		float* total_of_vals;
		cudaMalloc(&total_of_vals, sizeof(float));	CUERR("Allocate memory for total");
		dim3 grid(DIV_ROUNDUP(num_values, CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		warpReduceSum_kernel<<<grid, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(device_vals, num_values, total_of_vals);

		float* return_total = (float*)malloc(sizeof(float));
		cudaMemcpy(return_total, total_of_vals, sizeof(float), cudaMemcpyDeviceToHost);		CUERR("Copy total from device to host");

		REQUIRE( return_total[0] == 528 );

		cudaFree(device_vals);	      CUERR("Free values");
		cudaFree(total_of_vals);      CUERR("Free total");

		free(host_vals);
		free(return_total);

		std::cerr << std::endl;
	}
}

// int warpReduceMin(int val);
TEST_CASE(" Warp Reduce Min "){
	SECTION("Good Data"){
		std::cerr << "------TEST WARP REDUCE MIN GOOD DATA------" << std::endl;

		int num_values = 32;
		float* host_vals = (float*)malloc(sizeof(float)*num_values);
		for(int i = 0; i < num_values; i++){
			host_vals[i] = num_values - i;
		}

		float* device_vals;
		cudaMalloc(&device_vals, sizeof(float)*num_values);		CUERR("Allocate memory for values");
		cudaMemcpy(device_vals, host_vals, sizeof(float)*num_values, cudaMemcpyHostToDevice);	CUERR("Copy values from host to device");

		float* min_of_vals;
		cudaMalloc(&min_of_vals, sizeof(float));	CUERR("Allocate memory for total");
		dim3 grid(DIV_ROUNDUP(num_values, CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		warpReduceMin_kernel<<<grid, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(device_vals, num_values, min_of_vals);

		float* return_min = (float*)malloc(sizeof(float));
		cudaMemcpy(return_min, min_of_vals, sizeof(float), cudaMemcpyDeviceToHost);		CUERR("Copy total from device to host");

		REQUIRE( return_min[0] == 1 );

		cudaFree(device_vals);	      CUERR("Free values");
		cudaFree(min_of_vals);      CUERR("Free min");

		free(host_vals);
		free(return_min);

		std::cerr << std::endl;
	}
}

// void block_findMin(QTYPE *dtw_results, long long *idata, const int tid, const int block_dim, const long long strided_jobid, QTYPE_ACC &min, long long &index);
TEST_CASE(" Block Find Min "){

	long num_query_indices = CUDA_THREADBLOCK_MAX_THREADS;
	long existing_subject_length = CUDA_THREADBLOCK_MAX_THREADS*5;

	int threadblock_size_dtw = CUDA_THREADBLOCK_MAX_THREADS/MINIDTW_STRIDE;
	dim3 griddim_dtw(DIV_ROUNDUP(existing_subject_length, CUDA_THREADBLOCK_MAX_THREADS), 1, 1);

	int dtw_dist_size = existing_subject_length;

	QTYPE_ACC *dtw_dists = (QTYPE_ACC*)malloc(sizeof(QTYPE_ACC)*dtw_dist_size);

	QTYPE* dtw_results_device;
	cudaMalloc(&dtw_results_device, sizeof(QTYPE)*dtw_dist_size);

	QTYPE_ACC* min_device;
	cudaMalloc(&min_device, 5*sizeof(QTYPE_ACC));

	long long* index_device;
	cudaMalloc(&index_device, 5*sizeof(long long));

	QTYPE_ACC* min_host = (QTYPE_ACC*)malloc(5*sizeof(QTYPE_ACC));
	long long* index_host = (long long*)malloc(5*sizeof(long long));

	SECTION("Good Data Acending"){
		std::cerr << "------TEST BLOCK FIND MIN GOOD DATA ACENDING------" << std::endl;

		int counter = 0;
		for(int i = 0; i < dtw_dist_size; i++){
			dtw_dists[i] = counter + (i / 256);
			if(++counter == 256) counter = 0;
		}

		cudaMemcpy(dtw_results_device, dtw_dists, sizeof(QTYPE)*dtw_dist_size, cudaMemcpyHostToDevice);

		block_findMin_kernel<<<griddim_dtw, threadblock_size_dtw, 0, 0>>>(dtw_results_device, num_query_indices, min_device, index_device);

		cudaMemcpy(min_host, min_device, 5*sizeof(QTYPE_ACC), cudaMemcpyDeviceToHost);
		cudaMemcpy(index_host, index_device, 5*sizeof(long long), cudaMemcpyDeviceToHost);

		REQUIRE( round_to_three(min_host[0]) == (QTYPE_ACC)0 );
		REQUIRE( round_to_three(min_host[1]) == (QTYPE_ACC)1 );
		REQUIRE( index_host[0] == 0 );
		REQUIRE( index_host[1] == 256 );
		std::cerr << std::endl;
	}

	SECTION("Good Data Decending"){
		std::cerr << "------TEST BLOCK FIND MIN GOOD DATA DECENDING------" << std::endl;

		int counter = 0;
		for(int i = 0; i < dtw_dist_size; i++){
			dtw_dists[i] = 255 + (i / 256) - counter;
			if(++counter == 256) counter = 0;
		}
		std::cerr << std::endl;

		cudaMemcpy(dtw_results_device, dtw_dists, sizeof(QTYPE)*dtw_dist_size, cudaMemcpyHostToDevice);

		block_findMin_kernel<<<griddim_dtw, threadblock_size_dtw, 0, 0>>>(dtw_results_device, num_query_indices, min_device, index_device);

		cudaMemcpy(min_host, min_device, 5*sizeof(QTYPE_ACC), cudaMemcpyDeviceToHost);
		cudaMemcpy(index_host, index_device, 5*sizeof(long long), cudaMemcpyDeviceToHost);

		REQUIRE( round_to_three(min_host[0]) == (QTYPE_ACC)0 );
		REQUIRE( round_to_three(min_host[1]) == (QTYPE_ACC)1 );
		REQUIRE( index_host[0] == 255 );
		REQUIRE( index_host[1] == 511 );
		std::cerr << std::endl;
	}

	free(dtw_dists);
	free(min_host);
	free(index_host);

	cudaFree(dtw_results_device);
	cudaFree(min_device);
	cudaFree(index_device);
}

// void query_znorm(int offset, int length, float *data_mean, float *data_stddev, QTYPE *min, QTYPE *max, int normalization_mode, cudaStream_t stream=0);
TEST_CASE(" Query Znorm "){
	int sub_length = 100;
	QTYPE* subject = (QTYPE*)malloc(sizeof(QTYPE)*sub_length);
	for(int i = 0; i < sub_length; i++){
		subject[i] = i+1;
	}
	load_subject(subject, 0, sub_length, false);
	free(subject);

	float *mean, *stddev;
    QTYPE *min, *max;
    cudaMalloc(&mean, sizeof(float)); CUERR("Allocating GPU memory for query mean");
    cudaMalloc(&stddev, sizeof(float)); CUERR("Allocating GPU memory for query std dev");
    cudaMalloc(&min, sizeof(QTYPE)); CUERR("Allocating GPU memory for query min");
    cudaMalloc(&max, sizeof(QTYPE));CUERR("Allocating GPU memory for query max");

	int total_expected_segments = 20;
	QTYPE *query_elems = (QTYPE*)malloc(sizeof(QTYPE)*total_expected_segments);

	for(int i = 0; i < total_expected_segments; i++){
		if(i < total_expected_segments/2){
			query_elems[i] = i+1; 
		} else{
			query_elems[i] = total_expected_segments+i+1; 
		}
	}

	float tmp_mean = 18;
	float tmp_stddev = 4.899;
	QTYPE tmp_min = 1;
	QTYPE tmp_max = 40;
	
	SECTION("Good Data Global Znorm"){
		std::cerr << "------TEST QUERY ZNORM GOOD DATA GLOBAL ZNORM------" << std::endl;
		load_G_query<QTYPE>(query_elems, total_expected_segments);

		cudaMemcpy(mean, &tmp_mean, sizeof(float), cudaMemcpyHostToDevice);        CUERR("Copying mean from host to device");
		cudaMemcpy(stddev, &tmp_stddev, sizeof(float), cudaMemcpyHostToDevice);    CUERR("Copying stddev from host to device");
		cudaMemcpy(min, &tmp_min, sizeof(QTYPE), cudaMemcpyHostToDevice);          CUERR("Copying min from host to device");
		cudaMemcpy(max, &tmp_max, sizeof(QTYPE), cudaMemcpyHostToDevice);			CUERR("Copying max from host to device");

		dim3 norm_grid(DIV_ROUNDUP(total_expected_segments,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);

		query_znorm<QTYPE><<<norm_grid, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(0, total_expected_segments, mean, stddev, min, max, GLOBAL_ZNORM);		CUERR("Applying global query normalization");

		QTYPE *return_Gquery;
		int return_length;
		get_G_query(&return_Gquery, &return_length);
		
		REQUIRE( round_to_three(return_Gquery[0]) == -49.668f );
		REQUIRE( round_to_three(return_Gquery[total_expected_segments-1]) == 180.129f );

		free(return_Gquery);

		std::cerr << std::endl;
	}

	SECTION("Good Data Local Znorm"){
		std::cerr << "------TEST QUERY ZNORM GOOD DATA LOCAL ZNORM------" << std::endl;
		load_G_query<QTYPE>(query_elems, total_expected_segments);
		int num_nonzero_series = 2;

		float tmp_means[] = { 5.5, 35.5 };
		float tmp_stddevs[] = { 2.872, 2.872 };
		QTYPE tmp_mins[] = { 1, 31 };
		QTYPE tmp_maxs[] = { 10, 40 };

		int *nonzero_series_lengths;
		cudaMallocHost(&nonzero_series_lengths, sizeof(int)*num_nonzero_series);           CUERR("Allocating host memory for query lengths");

		int query_processed_so_far = 0;
		// Not really worth parallelizing this outer loop as the size of the query is very constrained for now
		for(int i = 0; i < num_nonzero_series; i++){
			nonzero_series_lengths[i] = 10;
			dim3 grid(DIV_ROUNDUP(nonzero_series_lengths[i],CUDA_THREADBLOCK_MAX_THREADS), 1, 1);

			cudaMemcpy(mean, &tmp_means[i], sizeof(float), cudaMemcpyHostToDevice);        CUERR("Copying mean from host to device");
            cudaMemcpy(stddev, &tmp_stddevs[i], sizeof(float), cudaMemcpyHostToDevice);    CUERR("Copying stddev from host to device");
			cudaMemcpy(min, &tmp_mins[i], sizeof(QTYPE), cudaMemcpyHostToDevice);          CUERR("Copying min from host to device");
			cudaMemcpy(max, &tmp_maxs[i], sizeof(QTYPE), cudaMemcpyHostToDevice);			CUERR("Copying max from host to device");

			query_znorm<QTYPE><<<grid, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(query_processed_so_far, nonzero_series_lengths[i], mean, stddev, min, max, LOCAL_ZNORM);	CUERR("Applying local query normalization");
			query_processed_so_far += nonzero_series_lengths[i];
		}

		QTYPE *return_Gquery;
		int return_length;
		get_G_query(&return_Gquery, &return_length);

		REQUIRE( round_to_three(return_Gquery[0]) == 5.271f );
		REQUIRE( round_to_three(return_Gquery[nonzero_series_lengths[0]-1]) == 95.729f );
		REQUIRE( round_to_three(return_Gquery[0]) == 5.271f );
		REQUIRE( round_to_three(return_Gquery[total_expected_segments-1]) == 95.729f );

		cudaFreeHost(nonzero_series_lengths);

		free(return_Gquery);

		std::cerr << std::endl;
	}

	SECTION("Good Data No Znorm"){
		std::cerr << "------TEST QUERY ZNORM GOOD DATA NO ZNORM------" << std::endl;
		load_G_query<QTYPE>(query_elems, total_expected_segments);
		dim3 norm_grid(DIV_ROUNDUP(total_expected_segments,CUDA_THREADBLOCK_MAX_THREADS), 1, 1);
		// Next two line used so the query location can be passed around like any other memory pointer to the stats function.

		cudaMemcpy(mean, &tmp_mean, sizeof(float), cudaMemcpyHostToDevice);        CUERR("Copying mean from host to device");
		cudaMemcpy(stddev, &tmp_stddev, sizeof(float), cudaMemcpyHostToDevice);    CUERR("Copying stddev from host to device");
		cudaMemcpy(min, &tmp_min, sizeof(QTYPE), cudaMemcpyHostToDevice);          CUERR("Copying min from host to device");
		cudaMemcpy(max, &tmp_max, sizeof(QTYPE), cudaMemcpyHostToDevice);			CUERR("Copying max from host to device");
		
		query_znorm<QTYPE><<<norm_grid, CUDA_THREADBLOCK_MAX_THREADS, 0, 0>>>(0, total_expected_segments, mean, stddev, min, max, NO_ZNORM);		CUERR("Applying no query normalization");
		cudaStreamSynchronize(0);				CUERR("Synchronizing stream after query znorm");

		QTYPE *return_Gquery;
		int return_length;
		get_G_query(&return_Gquery, &return_length);

		REQUIRE( return_Gquery[0] == 1 );
		REQUIRE( return_Gquery[total_expected_segments-1] == 40 );

		free(return_Gquery);

		std::cerr << std::endl;
	}
	
	cudaFree(mean);           CUERR("Free mean");
	cudaFree(stddev);         CUERR("Free stddev");
	cudaFree(min);            CUERR("Free min");
	cudaFree(max);            CUERR("Free max");

}