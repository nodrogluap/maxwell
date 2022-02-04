#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../ont_utils.cuh"
#include "../all_utils.hpp"

#include <string>
#include <vector>

#include <stdio.h>  /* defines FILENAME_MAX */
#if defined(_WIN32)
	#include <direct.h>
	#define GetCurrentDir _getcwd
	#define ONWINDOWS 1
#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#define ONWINDOWS 0
#endif
#include<iostream>

#define QTYPE float 
#define QTYPE_ACC float

// int convert_rna_to_shorts(char *cpu_mem_rna, long rna_length, short signal_type, short strand_types, short **global_gpu_mem_output_signal, long *signal_length, cudaStream_t stream=0)
TEST_CASE( " int Convert RNA to Shorts " ) {


	SECTION ("Good file name") {
		
	}
	
}

// int convert_dna_to_shorts(char *cpu_mem_dna, long dna_length, short signal_type, short strand_types, short **global_gpu_mem_output_signal, long *signal_length, cudaStream_t stream=0)
TEST_CASE( " int Convert DNA to Shorts " ) {


	SECTION ("Good file name") {
		
	}
	
}

// short* convert_rna_to_shorts(char *cpu_mem_rna, long rna_length, short signal_type, short strand_types, long *signal_length)
TEST_CASE( " short Convert RNA to Shorts " ) {


	SECTION ("Good file name") {
		
	}
	
}

// short* convert_dna_to_shorts(char *cpu_mem_dna, long dna_length, short signal_type, short strand_types, long *signal_length)
TEST_CASE( " short Convert DNA to Shorts " ) {


	SECTION ("Good file name") {
		
	}
	
}