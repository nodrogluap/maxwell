#ifndef __FLASH_DTW_UTILS
#define __FLASH_DTW_UTILS

#include "flash_dtw.cuh"

// Here are additional functions made for testing purposes
__host__ void get_subject_from_GPU(QTYPE **subject, long* subject_length, float* mean, float* stddev, QTYPE* min, QTYPE* max, cudaStream_t stream=0);

template <class T>
void get_G_query(T** return_Gquery, int* query_length);
template <class T>
void get_G_query_std(T** return_Gquery_std, int query_std_length);

template <class T>
void load_G_query(T* tmp_Gquery, int query_length);
template <class T>
void load_G_query_std(T* tmp_Gquery_std, int query_std_length);

// Host side function that just makes sure the subject was loaded properly
// subject - the subject buffer that will be populated with the data stored on the GPU
// subject_length - the length of the subject to be obtained from the GPU
// mean - the mean of the subject to be obtained from the GPU
// stddev - the standard deviation of the subject to be obtained from the GPU
// min - the minimum value of the subject to be obtained from the GPU
// max - the maximum value of the subject to be obtained from the GPU
// stream - cuda stream
__host__
void 
get_subject_from_GPU(QTYPE **subject, long* subject_length, float* mean, float* stddev, QTYPE* min, QTYPE* max, cudaStream_t stream){

	cudaMemcpyFromSymbol(subject_length, ::Tsubject_length,  sizeof(long)); 	CUERR("Copying subject length")
	(*subject) = (QTYPE*) malloc(sizeof(QTYPE)*(*subject_length));
	cudaMemcpy((*subject), Dsubject, sizeof(QTYPE)*(*subject_length), cudaMemcpyDeviceToHost); 			CUERR("Copying subject")
	cudaMemcpyFromSymbol(mean, ::Dsubject_mean, sizeof(float));			CUERR("Copying mean")
	cudaMemcpyFromSymbol(stddev, ::Dsubject_stddev, sizeof(float)); 		CUERR("Copying stddev")
	cudaMemcpyFromSymbol(min, ::Dsubject_min,  sizeof(QTYPE));				CUERR("Copying min")
	cudaMemcpyFromSymbol(max, ::Dsubject_max,  sizeof(QTYPE));				CUERR("Copying max")
}

// Host side function that verifies that Gquery was loaded properly
// return_Gquery - the buffer that will be populated with the Gquery data stored on the GPU
// query_length - the length of Gquery
template <class T>
__host__
void
get_G_query(T** return_Gquery, int* query_length){
	cudaMemcpyFromSymbol(query_length, ::Gquery_length, sizeof(int), 0, cudaMemcpyDeviceToHost);   CUERR("Checking for preloaded query in GPU memory");
	(*return_Gquery) = (T*) malloc(sizeof(T)*MAX_QUERY_SIZE);
	cudaMemcpyFromSymbol((*return_Gquery), Gquery, sizeof(T)*(*query_length)); 			CUERR("Copying Gquery")
}

// Host side function that verifies that Gquery_std was loaded properly
// return_Gquery_std - the buffer that will be populated with the Gquery_std data stored on the GPU
// query_std_length - the length of Gquery_std
template <class T>
__host__
void
get_G_query_std(T** return_Gquery_std, int query_std_length){
	(*return_Gquery_std) = (T*) malloc(sizeof(T)*MAX_QUERY_SIZE);
	cudaMemcpyFromSymbol((*return_Gquery_std), Gquery_std, sizeof(T)*query_std_length); 			CUERR("Copying Gquery_std")
}

// Host side function that verifies that loads query values into Gquery
// tmp_Gquery - the buffer that will populate Gquery
// query_length - the length of both Cquery and Gquery
template <class T>
__host__
void
load_G_query(T* tmp_Gquery, int query_length){
	cudaMemcpyToSymbol(Gquery, tmp_Gquery, sizeof(T)*query_length);                            CUERR("Copying Gquery from CPU to GPU emory")
	cudaMemcpyToSymbolAsync(::Gquery_length, &query_length, sizeof(int), 0, cudaMemcpyHostToDevice, 0);		CUERR("Copying query's length from CPU to GPU constant memory") 
}

// Host side function that verifies that loads query std values into Gquery_std
// tmp_Gquery_std - the buffer that will populate Gquery_std
// query_std_length - the length of both Cquery_std and Gquery_std
template <class T>
__host__
void
load_G_query_std(T* tmp_Gquery_std, int query_std_length){
	cudaMemcpyToSymbol(Gquery_std, tmp_Gquery_std, sizeof(T)*query_std_length);                            CUERR("Copying Gquery_std from CPU to GPU emory")
}

#endif
