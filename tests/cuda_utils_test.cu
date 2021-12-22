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

// TEST_CASE( " Bulk5 Data " ) {
	
// }

TEST_CASE( " Binary Data " ) {

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	std::string bad_file = current_working_dir + "/Not/a/real/path/binary_test.bin";
	std::string empty_file = current_working_dir + "/empty_files/empty.bin";

	QTYPE *query_values;
	size_t num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 10------" << std::endl;
		int result = read_binary_data<QTYPE>(good_file.c_str(), &query_values, &num_query_values);
	
		REQUIRE( result == 1 );
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

// TEST_CASE( " Read Sequence Bulk5 Files " ) {
	
// }

// TEST_CASE( " Read Sequence FastA Files " ) {
	
// }

// TEST_CASE( " Read Sequence Binary Files " ) {
	
// }

// TEST_CASE( " Read Sequence TSV Files " ) {
	
// }

// TEST_CASE( " Merge Data " ) {
	
// }

// TEST_CASE( " Data " ) {
	
// }