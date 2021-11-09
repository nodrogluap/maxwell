#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#if defined(_WIN32)
	#define GetCurrentDir _getcwd
	#define ONWINDOWS 1
#else
	#define GetCurrentDir getcwd
	#define ONWINDOWS 0
#endif
// Functions to read in data files (non-GPU based) 
#include "../flash_utils.hpp"
#define QTYPE float 
#define QTYPE_ACC float

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

TEST_CASE( " Binary Data " ) {

	std::string good_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin0";
	std::string bad_file = current_working_dir + "/Not/a/real/path/binary_test.bin";
	std::string empty_file = current_working_dir + "/empty_files/empty.bin";

	QTYPE *query_values;
	unsigned long long int num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 1------" << std::endl;
		int result = read_binary_data<QTYPE>(good_file.c_str(), &query_values, &num_query_values);
	
		REQUIRE( result == 0 );
		REQUIRE( num_query_values == 479 );

		// for(int i = 0; i < num_query_values; i++){
			// std::cerr << query_values[i] << " ";
		// }

		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 2------" << std::endl;
		int result = read_binary_data<QTYPE>(bad_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 1 );

		std::cerr << std::endl;
	}

	SECTION ("Empty file name") {
		std::cerr << "------TEST 3------" << std::endl;
		int result = read_binary_data<QTYPE>(empty_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 1 );

		std::cerr << std::endl;
	}
}


TEST_CASE( " Text Data " ) {

	std::string good_file = current_working_dir + "/good_files/text/good.txt";
	std::string bad_file = current_working_dir + "/Not/a/real/path/text_test.txt";
	std::string empty_file = current_working_dir + "/empty_files/empty.txt";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong.txt";

	QTYPE *query_values;
	unsigned long long int num_query_values;

	SECTION ("Good file name") {
		std::cerr << "------TEST 4------" << std::endl;
		int result = read_text_data<QTYPE>(good_file.c_str(), &query_values, &num_query_values);
	
		REQUIRE( result == 0 );
		REQUIRE( num_query_values == 483 );


		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {
		std::cerr << "------TEST 5------" << std::endl;
		int result = read_text_data<QTYPE>(bad_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 1 );

		std::cerr << std::endl;
	}
	SECTION ("Empty file name") {
		std::cerr << "------TEST 6------" << std::endl;
		int result = read_text_data<QTYPE>(empty_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 1 );

		std::cerr << std::endl;
	}
	SECTION ("Wrong file name") {
		std::cerr << "------TEST 7------" << std::endl;
		int result = read_text_data<QTYPE>(wrong_file.c_str(), &query_values, &num_query_values);

		REQUIRE( result == 1 );

		std::cerr << std::endl;
	}
}