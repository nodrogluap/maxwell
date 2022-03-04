#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../ont_utils.cuh"
// #include "../all_utils.hpp"

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

char input[] = { 'C', 'C', 'A', 'G', 'T', 'T', 'G', 'T', 'G', 'T', 'G', 'T', 'A', 'C', 'T', 'A', 'C', 'A' };
long input_length = 18;

// int convert_rna_to_shorts(char *cpu_mem_rna, long rna_length, short signal_type, short strand_types, short **global_gpu_mem_output_signal, long *signal_length, cudaStream_t stream=0)
TEST_CASE( " int Convert RNA to Shorts " ) {


	SECTION ("Forward data Mean") {
		
		std::cerr << "------TEST 1------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( cpu_mem_output_signal[0] == 1093 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Forward data stddev1") {
		
		std::cerr << "------TEST 2------" << std::endl;
		
		short signal_type = 1;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( cpu_mem_output_signal[0] == 32 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Forward data stddev2") {
		
		std::cerr << "------TEST 3------" << std::endl;
		
		short signal_type = 2;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( cpu_mem_output_signal[0] == 40 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Forward data stddev3") {
		
		std::cerr << "------TEST 4------" << std::endl;
		
		short signal_type = 3;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( cpu_mem_output_signal[0] == 73 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Compliment data") {
		
		std::cerr << "------TEST 5------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 2;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( cpu_mem_output_signal[13] == 1048 );
		REQUIRE( cpu_mem_output_signal[0] == 711 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Both data") {
		
		std::cerr << "------TEST 6------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 3;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");

		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 28 );
		REQUIRE( cpu_mem_output_signal[0] == 1093 );
		REQUIRE( cpu_mem_output_signal[14] == 711 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
}

// int convert_dna_to_shorts(char *cpu_mem_dna, long dna_length, short signal_type, short strand_types, short **global_gpu_mem_output_signal, long *signal_length, cudaStream_t stream=0)
TEST_CASE( " int Convert DNA to Shorts " ) {


	SECTION ("Forward data Mean") {
		
		std::cerr << "------TEST 7------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( cpu_mem_output_signal[0] == 782 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
	SECTION ("Forward data stddev1") {
		
		std::cerr << "------TEST 8------" << std::endl;
		
		short signal_type = 1;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( cpu_mem_output_signal[0] == 24 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
		
	SECTION ("Forward data stddev2") {
		
		std::cerr << "------TEST 9------" << std::endl;
		
		short signal_type = 2;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( cpu_mem_output_signal[0] == 18 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
		
	SECTION ("Forward data stddev3") {
		
		std::cerr << "------TEST 10------" << std::endl;
		
		short signal_type = 3;
		short strand_flags = 1;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( cpu_mem_output_signal[0] == 42 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
		
	SECTION ("Compliment data") {
		
		std::cerr << "------TEST 11------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 2;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( cpu_mem_output_signal[12] == 973 );
		REQUIRE( cpu_mem_output_signal[0] == 1034);
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
		
	SECTION ("Both data") {
		
		std::cerr << "------TEST 12------" << std::endl;
		
		short signal_type = 0;
		short strand_flags = 3;
		
		short* tmp_vals = 0;
		long num_seqs_this_input = 0;
		
		int result = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &tmp_vals, &num_seqs_this_input);
		
		short *cpu_mem_output_signal = (short *) std::malloc(sizeof(short)*num_seqs_this_input);
		cudaMemcpy(cpu_mem_output_signal, tmp_vals, sizeof(short)*num_seqs_this_input, cudaMemcpyDeviceToHost); CUERR("Copying short representation of RNA from GPU to CPU");
		
		REQUIRE( result == 0 );
		REQUIRE( num_seqs_this_input == 26 );
		REQUIRE( cpu_mem_output_signal[0] == 782 );
		REQUIRE( cpu_mem_output_signal[13] == 1034 );
		
		free(cpu_mem_output_signal);
		cudaFree(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
}

// short* convert_rna_to_shorts(char *cpu_mem_rna, long rna_length, short signal_type, short strand_types, long *signal_length)
// ONly need one of each of the following tests since the above tests basically test the same functionality
TEST_CASE( " short Convert RNA to Shorts " ) {


	SECTION ("Forward data Mean") {
		
		std::cerr << "------TEST 13------" << std::endl;

		short signal_type = 0;
		short strand_flags = 1;
		
		long num_seqs_this_input = 0;
		
		short* tmp_vals  = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_input);
		
		REQUIRE( num_seqs_this_input == 14 );
		REQUIRE( tmp_vals[0] == 1093 );
		
		free(tmp_vals);
		
		std::cerr << std::endl;
		
	}

	
}

// short* convert_dna_to_shorts(char *cpu_mem_dna, long dna_length, short signal_type, short strand_types, long *signal_length)
TEST_CASE( " short Convert DNA to Shorts " ) {


	SECTION ("Forward data Mean") {
		
		std::cerr << "------TEST 14------" << std::endl;
		

		short signal_type = 0;
		short strand_flags = 1;
		
		long num_seqs_this_input = 0;
		
		short* tmp_vals  = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_input);
		
		REQUIRE( num_seqs_this_input == 13 );
		REQUIRE( tmp_vals[0] == 782 );
		
		free(tmp_vals);
		
		std::cerr << std::endl;
		
	}
	
}