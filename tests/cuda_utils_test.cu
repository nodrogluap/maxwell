#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../cuda_utils.hpp"
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

#include "cuda_utils_test_utils.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

TEST_CASE( " Text Data " ) {

	std::string good_file = current_working_dir + "/good_files/text/good_test_query.txt";
	std::string bad_file = current_working_dir + "/Not/a/real/path/text_test.txt";
	std::string empty_file = current_working_dir + "/empty_files/empty.txt";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong.txt";

	QTYPE *query_values;
	size_t num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 1------" << std::endl;
		int result = read_text_data<QTYPE>(good_file.c_str(), &query_values, &num_query_values);
	
		REQUIRE( result == 1 );
		REQUIRE( num_query_values == 200 );


		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 2------" << std::endl;
		int result = read_text_data<QTYPE>(bad_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	SECTION ("Empty file name") {
		std::cerr << "------TEST 3------" << std::endl;
		int result = read_text_data<QTYPE>(empty_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	SECTION ("Wrong file name") {
		std::cerr << "------TEST 4------" << std::endl;
		int result = read_text_data<QTYPE>(wrong_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " Fast5 Data " ) {
	
	std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
	std::string vbz_file = current_working_dir + "/good_files/fast5/FAN41461_pass_496845aa_0.fast5";
	std::string bad_file = current_working_dir + "/Not/a/real/path/text_test.fast5";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fast5";

	QTYPE **query_values;
	char **query_names;
	size_t *num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 5------" << std::endl;
		
		cudaMallocHost(&query_values, sizeof(QTYPE*)*1);			CUERR("Allocating query_values on host");
		cudaMallocHost(&query_names, sizeof(char*)*1);				CUERR("Allocating query_names on host");
		cudaMallocHost(&num_query_values, sizeof(size_t));			CUERR("Allocating num_query_values on host");
		
		int result = read_fast5_data<QTYPE>(good_file.c_str(), query_values, query_names, num_query_values);
		
		std::string s_query_names = query_names[0];
	
		REQUIRE( result == 1 );
		REQUIRE( num_query_values[0] == 53789 );
		REQUIRE( s_query_names == "Read_352" );
		
		cudaFree(query_values[0]);			CUERR("Free query_values[0] on host");
		cudaFreeHost(query_names[0]);			CUERR("Free query_names[0] on host");
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");

		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 6------" << std::endl;
		
		int result = read_fast5_data<QTYPE>(bad_file.c_str(), query_values, query_names, num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	SECTION ("Empty file name") {
		std::cerr << "------TEST 7------" << std::endl;
		int result = read_fast5_data<QTYPE>(empty_file.c_str(), query_values, query_names, num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	SECTION ("Wrong file name") {
		std::cerr << "------TEST 8------" << std::endl;
		int result = read_fast5_data<QTYPE>(wrong_file.c_str(), query_values, query_names, num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	SECTION ("VBZ file") {
		std::cerr << "------TEST 9------" << std::endl;
		
		static int expected_num_reads = 11;
		
		cudaMallocHost(&query_values, sizeof(QTYPE*)*expected_num_reads);			CUERR("Allocating query_values on host");
		cudaMallocHost(&query_names, sizeof(char*)*expected_num_reads);				CUERR("Allocating query_names on host");
		cudaMallocHost(&num_query_values, sizeof(size_t));							CUERR("Allocating num_query_values on host");
		
		int result = read_fast5_data<QTYPE>(vbz_file.c_str(), query_values, query_names, num_query_values);
		
		std::cerr << "Num reads: " << num_query_values[0] << " and name: " << query_names[0] << std::endl;

		REQUIRE( result == expected_num_reads );
		// REQUIRE( num_query_values == 3141 );
		// REQUIRE( query_names == "Read_352" );
		
		for(int i = 0; i < expected_num_reads; i++){
			cudaFree(query_values[i]);			CUERR("Free a query_value on host");
			cudaFreeHost(query_names[i]);		CUERR("Free a query_name on host");
		}
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");

		std::cerr << std::endl;
	}
	
}

// TODO: Get a small bulk5 file to test this
// TEST_CASE( " Bulk5 Data " ) {
	
// }

TEST_CASE( " Binary Data " ) {

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	std::string bad_file = current_working_dir + "/Not/a/real/path/binary_test.bin";
	std::string empty_file = current_working_dir + "/empty_files/binary.bin";

	QTYPE *query_values;
	size_t num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 10------" << std::endl;
		int result = read_binary_data<QTYPE>(good_file.c_str(), &query_values, &num_query_values);
	
		REQUIRE( result == 1 );
		REQUIRE( query_values[0] == 849 );
		REQUIRE( num_query_values == 479 );

		// for(int i = 0; i < num_query_values; i++){
			// std::cerr << query_values[i] << " ";
		// }

		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 11------" << std::endl;
		int result = read_binary_data<QTYPE>(bad_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Empty file name") {
		std::cerr << "------TEST 12------" << std::endl;
		int result = read_binary_data<QTYPE>(empty_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " TSV Data " ) {
	
	std::string good_file = current_working_dir + "/good_files/good_file.tsv";
	std::string bad_file = current_working_dir + "/Not/a/real/path/tsv_test.tsv";
	std::string empty_file = current_working_dir + "/empty_files/empty_file.tsv";
	
	QTYPE **query_values;
	char ** query_names;
	size_t *num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 13------" << std::endl;
		
		cudaMallocHost(&query_values, sizeof(QTYPE *)*2); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&query_names, sizeof(char *)*2); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&num_query_values, sizeof(size_t)*2); CUERR("Allocating CPU memory for sequence lengths");
		
		int result = read_tsv_data<QTYPE>(good_file.c_str(), query_values, query_names, num_query_values);
	
		REQUIRE( result == 2 );
		REQUIRE( query_values[0][0] == 15 );
		REQUIRE( std::string(query_names[0]) == "test1" );
		REQUIRE( num_query_values[0] == 25 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");

		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 14------" << std::endl;
		
		cudaMallocHost(&query_values, sizeof(QTYPE *)*2); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&query_names, sizeof(char *)*2); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&num_query_values, sizeof(size_t)*2); CUERR("Allocating CPU memory for sequence lengths");
		
		int result = read_tsv_data<QTYPE>(bad_file.c_str(), query_values, query_names, num_query_values);

		REQUIRE( result == 0 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");

		std::cerr << std::endl;
	}

	SECTION ("Empty file name") {
		std::cerr << "------TEST 15------" << std::endl;
		
		cudaMallocHost(&query_values, sizeof(QTYPE *)*2); CUERR("Allocating CPU memory for sequence pointers");
		cudaMallocHost(&query_names, sizeof(char *)*2); CUERR("Allocating CPU memory for sequence lengths");
		cudaMallocHost(&num_query_values, sizeof(size_t)*2); CUERR("Allocating CPU memory for sequence lengths");
		
		int result = read_tsv_data<QTYPE>(empty_file.c_str(), query_values, query_names, num_query_values);

		REQUIRE( result == 0 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");

		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Read Sequence Text Files " ) {
	
	char** filenames = (char**)malloc(sizeof(char*)*3);
	
	std::string good_file = current_working_dir + "/good_files/text/good_test_query.txt";
	std::string bad_file = current_working_dir + "/Not/a/real/path/text_test.txt";
	std::string empty_file = current_working_dir + "/empty_files/empty.txt";
	
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(bad_file);
	filenames[2] = stringToChar(empty_file);
	
	SECTION ("Good file name") {
		
		std::cerr << "------TEST 16------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		
		stringToChar(good_file);
		result = readSequenceTextFiles(&filenames[0], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == 1 );
		REQUIRE( *num_query_values == 200 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");
	
	}
	
	SECTION ("Bad file name") {
		
		std::cerr << "------TEST 17------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		result = readSequenceTextFiles(&filenames[1], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == 0 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");
	
	}
	
	SECTION ("Empty file name") {
		
		std::cerr << "------TEST 18------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		result = readSequenceTextFiles(&filenames[2], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == 0 );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");
	
	}
	
	free(filenames);
}

TEST_CASE( " Read Sequence Fast5 Files " ) {
	
	char** filenames = (char**)malloc(sizeof(char*)*4);
	
	std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
	std::string vbz_file = current_working_dir + "/good_files/fast5/FAN41461_pass_496845aa_0.fast5";
	std::string bad_file = current_working_dir + "/Not/a/real/path/text_test.fast5";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
	
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(vbz_file);
	filenames[2] = stringToChar(bad_file);
	filenames[3] = stringToChar(empty_file);
	
	SECTION ("Good file name") {
		
		std::cerr << "------TEST 19------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		
		stringToChar(good_file);
		result = readSequenceFAST5Files(&filenames[0], 1, &query_values, &query_names, &num_query_values);
		
		std::string s_query_names = query_names[0];
		
		REQUIRE( result == 1 );
		REQUIRE( num_query_values[0] == 53789 );
		REQUIRE( s_query_names == "Read_352" );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");
	
	}
	
	SECTION ("VBZ name") {
		
		std::cerr << "------TEST 20------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		
		static int expected_num_reads = 11;
		
		stringToChar(good_file);
		result = readSequenceFAST5Files(&filenames[1], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == expected_num_reads );
		// REQUIRE( num_query_values[0] == 53789 );
		// REQUIRE( s_query_names == "Read_352" );
		
		cudaFreeHost(query_values);				CUERR("Free query_values on host");
		cudaFreeHost(query_names);				CUERR("Free query_names on host");
		cudaFreeHost(num_query_values);			CUERR("Free num_query_values on host");
	
	}
	
	SECTION ("Bad file name") {
		
		std::cerr << "------TEST 21------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		result = readSequenceFAST5Files(&filenames[2], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == 0 );
		
	}
	
	SECTION ("Empty file name") {
		
		std::cerr << "------TEST 22------" << std::endl;
	
		QTYPE** query_values;
		char** query_names;
		size_t* num_query_values;
		size_t result = 0;
		result = readSequenceFAST5Files(&filenames[3], 1, &query_values, &query_names, &num_query_values);
		
		REQUIRE( result == 0 );
	
	}
	
	free(filenames);
	
}

// TODO: Get a small bulk5 file to test this
// TEST_CASE( " Read Sequence Bulk5 Files " ) {
	
// }


// TODO: Ask about this tomorrow. Should it return number of sequences or number of files
TEST_CASE( " Read Sequence FastA Files " ) {
	
	int num_files = 1;
	int rna = 0;
	short signal_type = 0;
	short strand_flags = 1;
	
	char** filenames = (char**)malloc(sizeof(char*)*4);
	
	std::string good_file = current_working_dir + "/good_files/fasta/fna/good_file.fna";	
	std::string bad_file = current_working_dir + "/nope/not/real.fna";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fna";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fna";
	
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(bad_file);
	filenames[2] = stringToChar(empty_file);
	filenames[3] = stringToChar(wrong_file);
	
	SECTION("Good file dna"){
		
		std::cerr << "------TEST 24------" << std::endl;
			
		
		QTYPE** sequences;
		char** sequence_names;
		size_t* sequence_lengths;
			
		int result = readSequenceFASTAFiles(&filenames[0], num_files, &sequences, &sequence_names, &sequence_lengths, rna, signal_type, strand_flags);
		
		std::string query_names_s1 = std::string(sequence_names[0]);
		std::string query_names_s2 = std::string(sequence_names[1]);
		
		REQUIRE( result == 6 );
		REQUIRE( query_names_s1 == ">chrI" );
		REQUIRE( query_names_s2 == ">chrII" );
		REQUIRE( sequences[0][0] == 782 );
		REQUIRE( sequence_lengths[0] == 13 );
		REQUIRE( sequence_lengths[1] == 13 );
		REQUIRE( sequence_lengths[2] == 5 );
		
		cudaFreeHost(sequences);			CUERR("Free sequences on host fasta");
		// for(int i = 0; i < num_files; i++){
			// cudaFreeHost(sequence_names[i]);	CUERR("Free seq name in fasta");
		// }
		cudaFreeHost(sequence_names);			CUERR("Free sequence_names on host fasta");
		cudaFreeHost(sequence_lengths);		CUERR("Free sequence_lengths on host fasta");
		
		std::cerr << std::endl;
	}
	
	rna = 1;
	
	SECTION("Good file rna"){
	
	std::cerr << "------TEST 25------" << std::endl;
			
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceFASTAFiles(&filenames[0], num_files, &sequences, &sequence_names, &sequence_lengths, rna, signal_type, strand_flags);
		
		std::string query_names_s1 = std::string(sequence_names[0]);
		std::string query_names_s2 = std::string(sequence_names[1]);
		
		REQUIRE( result == 6 );
		REQUIRE( query_names_s1 == ">chrI" );
		REQUIRE( query_names_s2 == ">chrII" );
		REQUIRE( sequences[0][0] == 1093 );
		REQUIRE( sequence_lengths[0] == 14 );
		REQUIRE( sequence_lengths[1] == 14 );
		REQUIRE( sequence_lengths[2] == 6 );
		
		cudaFreeHost(sequences);			CUERR("Free sequences on host fasta");
		// for(int i = 0; i < num_files; i++){
			// cudaFreeHost(sequence_names[i]);	CUERR("Free seq name in fasta");
		// }
		cudaFreeHost(sequence_names);			CUERR("Free sequence_names on host fasta");
		cudaFreeHost(sequence_lengths);		CUERR("Free sequence_lengths on host fasta");
		
		std::cerr << std::endl;
	}
	
	SECTION("Bad file"){
		
		std::cerr << "------TEST 26------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceFASTAFiles(&filenames[1], num_files, &sequences, &sequence_names, &sequence_lengths, rna, signal_type, strand_flags);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	SECTION("Empty file"){
		
		std::cerr << "------TEST 27------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceFASTAFiles(&filenames[2], num_files, &sequences, &sequence_names, &sequence_lengths, rna, signal_type, strand_flags);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	// TODO: Not sure how this should really run if some of the file contents are incorrect
	// SECTION("Wrong file"){
		
		// std::cerr << "------TEST 28------" << std::endl;
		
		// QTYPE **sequences;
		// char **sequence_names;
		// size_t *sequence_lengths;
			
		// int result = readSequenceFASTAFiles(&filenames[3], num_files, &sequences, &sequence_names, &sequence_lengths, rna, signal_type, strand_flags);
		
		// REQUIRE( result == 0 );
		
		// std::cerr << std::endl;
	// }

	free(filenames);
}

// int readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths)
TEST_CASE( " Read Sequence Binary Files " ) {
	
	int num_files = 1;
	
	char** filenames = (char**)malloc(sizeof(char*)*4);
	
	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin";	
	std::string bad_file = current_working_dir + "/nope/not/real.bin";
	std::string empty_file = current_working_dir + "/empty_files/binary.bin";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong.bin";
	
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(bad_file);
	filenames[2] = stringToChar(empty_file);
	filenames[3] = stringToChar(wrong_file);
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 28------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
		
		int result = readSequenceBinaryFiles(&filenames[0], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		std::string query_names_s1 = std::string(sequence_names[0]);
		
		REQUIRE( result == 1 );
		REQUIRE( query_names_s1 == "/gpfs/home/sjhepbur/maxwell/tests/good_files/binary/coxsackie_a16_3prime.fna.bin" );
		REQUIRE( sequences[0][0] == 849 );
		REQUIRE( sequence_lengths[0] == 479 );
		
		cudaFreeHost(sequences);			CUERR("Free sequences on host binary");
		cudaFreeHost(sequence_names);			CUERR("Free sequence_names on host binary");
		cudaFreeHost(sequence_lengths);		CUERR("Free sequence_lengths on host binary");
		
	}
	
	SECTION("Bad file"){
		
		std::cerr << "------TEST 29------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceBinaryFiles(&filenames[1], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	SECTION("Empty file"){
		
		std::cerr << "------TEST 30------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceBinaryFiles(&filenames[2], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	// TODO: Not sure how this should really run if some of the file contents are incorrect
	// SECTION("Wrong file"){
		
		// std::cerr << "------TEST 30------" << std::endl;
		
		// QTYPE **sequences;
		// char **sequence_names;
		// size_t *sequence_lengths;
			
		// int result = readSequenceBinaryFiles(&filenames[3], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		// REQUIRE( result == 0 );
		
		// std::cerr << std::endl;
	// }
	
	free(filenames);
	
}

// int readSequenceTSVFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths)
TEST_CASE( " Read Sequence TSV Files " ) {
	
	int num_files = 1;
	
	char** filenames = (char**)malloc(sizeof(char*)*4);
	
	std::string good_file = current_working_dir + "/good_files/good_file.tsv";	
	std::string bad_file = current_working_dir + "/nope/not/real.tsv";
	std::string empty_file = current_working_dir + "/empty_files/empty_file.tsv";
	// std::string wrong_file = current_working_dir + "/wrong_files/wrong.tev";
	
	filenames[0] = stringToChar(good_file);
	filenames[1] = stringToChar(bad_file);
	filenames[2] = stringToChar(empty_file);
	// filenames[3] = stringToChar(wrong_file);
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 31------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
		
		int result = readSequenceTSVFiles(&filenames[0], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		std::string query_names_s1 = std::string(sequence_names[0]);
		
		REQUIRE( result == 2 );
		REQUIRE( query_names_s1 == "test1" );
		REQUIRE( sequences[0][0] == 15 );
		REQUIRE( sequence_lengths[0] == 25 );
		
		cudaFreeHost(sequences);			CUERR("Free sequences on host binary");
		cudaFreeHost(sequence_names);			CUERR("Free sequence_names on host binary");
		cudaFreeHost(sequence_lengths);		CUERR("Free sequence_lengths on host binary");
		
	}
	
	SECTION("Bad file"){
		
		std::cerr << "------TEST 32------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceTSVFiles(&filenames[1], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	SECTION("Empty file"){
		
		std::cerr << "------TEST 33------" << std::endl;
		
		QTYPE **sequences;
		char **sequence_names;
		size_t *sequence_lengths;
			
		int result = readSequenceTSVFiles(&filenames[2], num_files, &sequences, &sequence_names, &sequence_lengths);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}
	
}

// merge_data(T **input_vals, size_t *num_input_vals, std::vector< std::pair<size_t, char*> >& subject_offsets, char** sequence_names, int num_seq, size_t *total_vals)
TEST_CASE( " Merge Data " ) {
	
	
	SECTION("Good Data"){
		
		std::cerr << "------TEST 34------" << std::endl;
		
		int num_seq = 3;
		
		size_t *num_input_vals = 0;
		cudaMallocHost(&num_input_vals, sizeof(size_t)*num_seq); CUERR("Allocating CPU memory for num_input_vals");
		num_input_vals[0] = 10;
		num_input_vals[1] = 15;
		num_input_vals[2] = 12;
		
		QTYPE **input_vals = 0;
		cudaMallocHost(&input_vals, sizeof(QTYPE*)*num_seq); CUERR("Allocating CPU memory for input_vals");
		cudaMallocHost(&(input_vals[0]), sizeof(QTYPE)*num_input_vals[0]); CUERR("Allocating CPU memory for input_vals[0]");
		cudaMallocHost(&(input_vals[1]), sizeof(QTYPE)*num_input_vals[1]); CUERR("Allocating CPU memory for input_vals[1]");
		cudaMallocHost(&(input_vals[2]), sizeof(QTYPE)*num_input_vals[2]); CUERR("Allocating CPU memory for input_vals[2]");
		
		for(int i = 0; i < num_seq; i++){
			for(int j = 0; j < num_input_vals[i]; j++){
				input_vals[i][j] = i * j;
			}
		}
		
		char** sequence_names = (char**)malloc(sizeof(char**)*num_seq);
		std::string name_1 = "test1";
		sequence_names[0] = (char*)malloc(sizeof(char)*name_1.length());
		strcpy(sequence_names[0], name_1.c_str());
		std::string name_2 = "test2";
		sequence_names[1] = (char*)malloc(sizeof(char)*name_2.length());
		strcpy(sequence_names[1], name_2.c_str());
		std::string name_3 = "test3";
		sequence_names[2] = (char*)malloc(sizeof(char)*name_3.length());
		strcpy(sequence_names[2], name_3.c_str());
		
		std::vector< std::pair<size_t, char*> > subject_offsets;
		size_t total_vals = 0;

		QTYPE* merged_data = merge_data(input_vals, num_input_vals, subject_offsets, sequence_names, num_seq, &total_vals);
		
		std::pair<size_t, char*> return_pair = subject_offsets.at(0);
		size_t return_size = std::get<0>(return_pair);
		char* return_char = std::get<1>(return_pair);
		std::string return_char_s = std::string(return_char);
		
		REQUIRE( return_size == 10 );
		REQUIRE( return_char_s == "test1" );
		REQUIRE( merged_data[9] == 0 );
		REQUIRE( merged_data[24] == 14 );
		REQUIRE( merged_data[36] == 22 );
		REQUIRE( total_vals == 37);

		for(int i = 0; i < num_seq; i++){
			free(sequence_names[i]);
		}
		free(sequence_names);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("No Data"){
		
		std::cerr << "------TEST 35------" << std::endl;
		
		int num_seq = 3;
		
		size_t *num_input_vals = 0;
		cudaMallocHost(&num_input_vals, sizeof(size_t)*num_seq); CUERR("Allocating CPU memory for num_input_vals");
		num_input_vals[0] = 0;
		num_input_vals[1] = 0;
		num_input_vals[2] = 0;
		
		QTYPE **input_vals = 0;
		
		char** sequence_names = (char**)malloc(sizeof(char**)*num_seq);
		std::string name_1 = "test1";
		sequence_names[0] = (char*)malloc(sizeof(char)*name_1.length());
		strcpy(sequence_names[0], name_1.c_str());
		std::string name_2 = "test2";
		sequence_names[1] = (char*)malloc(sizeof(char)*name_2.length());
		strcpy(sequence_names[1], name_2.c_str());
		std::string name_3 = "test3";
		sequence_names[2] = (char*)malloc(sizeof(char)*name_3.length());
		strcpy(sequence_names[2], name_3.c_str());
		
		std::vector< std::pair<size_t, char*> > subject_offsets;
		size_t total_vals = 0;

		QTYPE* merged_data = merge_data(input_vals, num_input_vals, subject_offsets, sequence_names, num_seq, &total_vals);

		REQUIRE( merged_data == 0 );
		REQUIRE( total_vals == 0);

		for(int i = 0; i < num_seq; i++){
			free(sequence_names[i]);
		}
		free(sequence_names);
		
		std::cerr << std::endl;
		
	}
	
	SECTION("No Data in One Seq"){
		
		std::cerr << "------TEST 36------" << std::endl;
		
		int num_seq = 3;
		
		size_t *num_input_vals = 0;
		cudaMallocHost(&num_input_vals, sizeof(size_t)*num_seq); CUERR("Allocating CPU memory for num_input_vals");
		num_input_vals[0] = 10;
		num_input_vals[1] = 0;
		num_input_vals[2] = 12;
		
		QTYPE **input_vals = 0;
		cudaMallocHost(&input_vals, sizeof(QTYPE*)*num_seq); CUERR("Allocating CPU memory for input_vals");
		cudaMallocHost(&(input_vals[0]), sizeof(QTYPE)*num_input_vals[0]); CUERR("Allocating CPU memory for input_vals[0]");
		// cudaMallocHost(&(input_vals[1]), sizeof(QTYPE)*num_input_vals[1]); CUERR("Allocating CPU memory for input_vals[1]");
		cudaMallocHost(&(input_vals[2]), sizeof(QTYPE)*num_input_vals[2]); CUERR("Allocating CPU memory for input_vals[2]");
		
		for(int i = 0; i < num_seq; i++){
			for(int j = 0; j < num_input_vals[i]; j++){
				input_vals[i][j] = i * j;
			}
		}
		
		char** sequence_names = (char**)malloc(sizeof(char**)*num_seq);
		std::string name_1 = "test1";
		sequence_names[0] = (char*)malloc(sizeof(char)*name_1.length());
		strcpy(sequence_names[0], name_1.c_str());
		std::string name_2 = "test2";
		sequence_names[1] = (char*)malloc(sizeof(char)*name_2.length());
		strcpy(sequence_names[1], name_2.c_str());
		std::string name_3 = "test3";
		sequence_names[2] = (char*)malloc(sizeof(char)*name_3.length());
		strcpy(sequence_names[2], name_3.c_str());
		
		std::vector< std::pair<size_t, char*> > subject_offsets;
		size_t total_vals = 0;

		QTYPE* merged_data = merge_data(input_vals, num_input_vals, subject_offsets, sequence_names, num_seq, &total_vals);
		
		std::pair<size_t, char*> return_pair = subject_offsets.at(0);
		size_t return_size = std::get<0>(return_pair);
		char* return_char = std::get<1>(return_pair);
		std::string return_char_s = std::string(return_char);
		
		REQUIRE( return_size == 10 );
		REQUIRE( return_char_s == "test1" );
		REQUIRE( merged_data[9] == 0 );
		REQUIRE( merged_data[21] == 22 );
		REQUIRE( total_vals == 22);

		for(int i = 0; i < num_seq; i++){
			free(sequence_names[i]);
		}
		free(sequence_names);
		
		std::cerr << std::endl;
		
	}
	
}

//int read_data(char **filenames, int num_files, T ***output_vals, char ***sequence_names, size_t **num_output_vals, int instrand, int rna, short signal_type, short strand_flags)
TEST_CASE( " Data " ) {
	
	int num_files = 1;
	int total_num_files = 4;
	
	char** filenames = (char**)malloc(sizeof(char*)*total_num_files);
	
	std::string text_file = current_working_dir + "/good_files/text/good_test_query.txt";
	std::string fast5_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
	std::string fasta_file = current_working_dir + "/good_files/fasta/fna/good_file.fna";
	std::string wrong_file = current_working_dir + "/not/a/real/file.none";
	//std::string empty_file = current_working_dir + "/empty_files/empty_file.tsv";
	
	filenames[0] = stringToChar(text_file);
	filenames[1] = stringToChar(fast5_file);
	filenames[2] = stringToChar(fasta_file);
	filenames[3] = stringToChar(wrong_file);
	// filenames[4] = stringToChar(empty_file);
	
	int instrand = 3;
	int rna = 0;
	int signal_type = MEAN_SIGNAL;
	short strand_flags = 1;
	
	SECTION("Text Data"){
		
		std::cerr << "------TEST 37------" << std::endl;
		
		QTYPE **sequence_values = 0;
		char **sequence_names = 0;
		size_t *num_sequence_values = 0;
		
		int num_sequences = read_data<QTYPE>(&filenames[0], num_files, &sequence_values, &sequence_names, &num_sequence_values, instrand, rna, signal_type, strand_flags);
		
		REQUIRE( num_sequences == 1 );
		REQUIRE( sequence_values[0][0] == 371 );
		REQUIRE( sequence_values[0][1] == 803 );
		REQUIRE( sequence_values[0][2] == 2409 );
		REQUIRE( num_sequence_values[0] == 200 );
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Fast5 Data"){
		
		std::cerr << "------TEST 38------" << std::endl;
		
		QTYPE **sequence_values = 0;
		char **sequence_names = 0;
		size_t *num_sequence_values = 0;
		
		int num_sequences = read_data<QTYPE>(&filenames[1], num_files, &sequence_values, &sequence_names, &num_sequence_values, instrand, rna, signal_type, strand_flags);
		
		std::string s_query_names = sequence_names[0];
		
		REQUIRE( num_sequences == 1 );
		REQUIRE( num_sequence_values[0] == 53789 );
		REQUIRE( s_query_names == "Read_352" );
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Fasta Data"){
		
		std::cerr << "------TEST 39------" << std::endl;
		
		QTYPE **sequence_values = 0;
		char **sequence_names = 0;
		size_t *num_sequence_values = 0;
		
		int num_sequences = read_data<QTYPE>(&filenames[2], num_files, &sequence_values, &sequence_names, &num_sequence_values, instrand, rna, signal_type, strand_flags);
		
		std::string query_names_s1 = std::string(sequence_names[0]);
		std::string query_names_s2 = std::string(sequence_names[1]);
		
		REQUIRE( num_sequences == 6 );
		REQUIRE( query_names_s1 == ">chrI" );
		REQUIRE( query_names_s2 == ">chrII" );
		REQUIRE( sequence_values[0][0] == 782 );
		REQUIRE( num_sequence_values[0] == 13 );
		REQUIRE( num_sequence_values[1] == 13 );
		REQUIRE( num_sequence_values[2] == 5 );
		
		std::cerr << std::endl;
		
	}
	
	SECTION("Wrong Data"){
		
		std::cerr << "------TEST 40------" << std::endl;
		
		QTYPE **sequence_values = 0;
		char **sequence_names = 0;
		size_t *num_sequence_values = 0;
		
		int num_sequences = read_data<QTYPE>(&filenames[3], num_files, &sequence_values, &sequence_names, &num_sequence_values, instrand, rna, signal_type, strand_flags);
		
		REQUIRE( num_sequences == 0 );
		
		std::cerr << std::endl;
		
	}
	
	SECTION("All Data"){
		
		std::cerr << "------TEST 41------" << std::endl;
		
		QTYPE **sequence_values = 0;
		char **sequence_names = 0;
		size_t *num_sequence_values = 0;
		
		int num_sequences = read_data<QTYPE>(filenames, total_num_files, &sequence_values, &sequence_names, &num_sequence_values, instrand, rna, signal_type, strand_flags);
		
		REQUIRE( num_sequences == 8 );
		REQUIRE( num_sequence_values[0] == 200 );
		REQUIRE( num_sequence_values[1] == 53789 );
		REQUIRE( num_sequence_values[2] == 13 );
		
		std::cerr << std::endl;
		
	}
	
	free(filenames);
	
}