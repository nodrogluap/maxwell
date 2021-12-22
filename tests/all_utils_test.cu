#define CATCH_CONFIG_MAIN
#include "catch.hpp"
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

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

TEST_CASE( " File Read " ) {
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 1------" << std::endl;
		std::string good_file = current_working_dir + "/good_files/text/good_test_query.txt";
		
		const int size = 1024 * 1024;
		std::vector <char> read_buffer( size );
		std::ifstream ifs(good_file); 
		
		int result = FileRead(ifs, read_buffer);
		
		REQUIRE( result == 1118 );
		
		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Count Lines " ) {
	
	SECTION("String with 1 line"){
		std::cerr << "------TEST 2------" << std::endl;
		std::string s = "Hello World!\n";
 
		std::vector<char> read_buffer(s.begin(), s.end());
		
		int result = CountLines(read_buffer, s.size());
		
		REQUIRE( result == 1 );
		
		std::cerr <<std::endl;
	}
	
	SECTION("String with 2 lines"){
		std::cerr << "------TEST 3------" << std::endl;
		std::string s = "Hello World!\n This is a test line of a string\n";
 
		std::vector<char> read_buffer(s.begin(), s.end());
		
		int result = CountLines(read_buffer, s.size());
		
		REQUIRE( result == 2 );
		
		std::cerr <<std::endl;
	}
	
	SECTION("String with no lines"){
		std::cerr << "------TEST 4------" << std::endl;
		std::string s = "";
 
		std::vector<char> read_buffer(s.begin(), s.end());
		
		int result = CountLines(read_buffer, s.size());
		
		REQUIRE( result == 0 );
		
		std::cerr <<std::endl;
	}
	
}

TEST_CASE( " Has Ending " ) {
	
	SECTION("Matching Ending"){
		
		std::cerr << "------TEST 5------" << std::endl;
		std::string test = "This is a test";
		std::string end = "test";
		
		bool result = hasEnding (test, end);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("Non-Matching Ending"){
		
		std::cerr << "------TEST 6------" << std::endl;
		std::string test = "This is a test";
		std::string end = "maxwell";
		
		bool result = hasEnding (test, end);
		
		REQUIRE( result == false );
		
		std::cerr <<std::endl;
	}
	
}

TEST_CASE( " Is Bulk5 " ) {
	
	// TODO: Get a small bulk5 file in repo to test this
	// SECTION("Is a Bulk5 File"){
		
		// std::string bulk5_file = current_working_dir + "/good_files/some/path/to/bulk5";
		// bool result = isBulkFast5(stringToChar(bulk5_file));
		
		// REQUIRE( result == true );
	// }
	
	SECTION("Is Not a Bulk5 File"){

		std::cerr << "------TEST 7------" << std::endl;
		std::string non_bulk5_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
		bool result = isBulkFast5(stringToChar(non_bulk5_file));
		
		REQUIRE( result == false );
		
		std::cerr <<std::endl;
	}
	
}

TEST_CASE( " Check Ending " ) {
	
	SECTION("fast5 File"){
		
		std::cerr << "------TEST 8------" << std::endl;
		std::string fast5_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
		bool result = checkEnding(fast5_file);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("txt File"){
		
		std::cerr << "------TEST 9------" << std::endl;
		std::string txt_file = current_working_dir + "/good_files/text/good_test_query.txt";
		bool result = checkEnding(txt_file);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("fna File"){
		
		std::cerr << "------TEST 10------" << std::endl;
		std::string fna_file = current_working_dir + "/good_files/fasta/fna/good_file.fna";
		bool result = checkEnding(fna_file);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("bin File"){
		
		std::cerr << "------TEST 11------" << std::endl;
		std::string bin_file = current_working_dir + "/good_files/binary/coxsackie_a16_3prime.fna.bin";
		bool result = checkEnding(bin_file);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("tsv File"){
		
		std::cerr << "------TEST 12------" << std::endl;
		std::string tsv_file = current_working_dir + "/good_files/good_file.tsv";
		bool result = checkEnding(tsv_file);
		
		REQUIRE( result == true );
		
		std::cerr << std::endl;
	}
	
	SECTION("Non-Matching File"){
		
		std::cerr << "------TEST 13------" << std::endl;
		std::string fast5_file = current_working_dir + "/good_files/good.bed";
		bool result = checkEnding(fast5_file);
		
		REQUIRE( result == false );
		
		std::cerr << std::endl;
	}

	
}

TEST_CASE( " Get All Files From Directory " ) {
	
	SECTION("Good Dir"){
		
		std::cerr << "------TEST 14------" << std::endl;
		std::string good_dir = current_working_dir + "/empty_files";
		
		char** all_files;
		
		int result = getAllFilesFromDir(stringToChar(good_dir), &all_files);
		
		REQUIRE( result == 5 );
		
		std::cerr << std::endl;
	}
	
	SECTION("Bad Dir"){
		
		std::cerr << "------TEST 15------" << std::endl;
		std::string bad_dir = current_working_dir + "/path/to/bad/dir";
		
		char** all_files;
		
		int result = getAllFilesFromDir(stringToChar(bad_dir), &all_files);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}
	
	SECTION("No Files in Dir"){
		
		std::cerr << "------TEST 16------" << std::endl;
		std::string no_files_dir = current_working_dir + "/UCR_Suite";
		
		char** all_files;
		
		int result = getAllFilesFromDir(stringToChar(no_files_dir), &all_files);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Populate Tree " ) {

	std::string good_file = current_working_dir + "/good_files/good.bed";
	std::string bad_file = current_working_dir + "/nope/not/real.bed";
	std::string empty_file = current_working_dir + "/empty_files/empty.bed";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong.bed";

	ITree::interval_vector bed_intervals;

	SECTION("Good file"){

		std::cerr << "------TEST 17------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(good_file), 1);

		REQUIRE( result == 1 );
		REQUIRE( bed_intervals.empty() == false );


		std::string name1 = "test";
		std::string name2 = "abb";
		std::string name3 = "jump";
		std::string name4 = "saf";
		std::string name5 = "ending";

		ITree bed_tree = ITree(std::move(bed_intervals));
		auto tree_results = bed_tree.findOverlapping(500, 900);

		REQUIRE( tree_results.size() == 2 );
		REQUIRE( tree_results.at(0).value == name1 );
		REQUIRE( tree_results.at(1).value == name2 );

		tree_results = bed_tree.findOverlapping(65500, 66500);

		REQUIRE( tree_results.size() == 3 );
		REQUIRE( tree_results.at(0).value == name4 );
		REQUIRE( tree_results.at(1).value == name3 );
		REQUIRE( tree_results.at(2).value == name5 );
		
		std::cerr << std::endl;
	}

	SECTION("Bad file"){

		std::cerr << "------TEST 18------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(bad_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 19------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(empty_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 20------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(wrong_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Scan Fast5 Data " ) {
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 21------" << std::endl;
		std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
		size_t num_sequences;
		
		int result = scan_fast5_data(good_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 1 );
		
		std::cerr << std::endl;
	}

	SECTION("Bad file"){
		
		std::cerr << "------TEST 22------" << std::endl;
		std::string bad_file = current_working_dir + "/nope/not/real.fast5";
		size_t num_sequences;
		
		int result = scan_fast5_data(bad_file.c_str(), &num_sequences);
		
		REQUIRE( result == FAST5_FILE_UNREADABLE );
		
		std::cerr << std::endl;
	}

	// TODO: See if we can get an empty fast5 file somehow
	// SECTION("Empty file"){
		
		// std::cerr << "------TEST 23------" << std::endl;
		// std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
		// size_t num_sequences;
		
		// int result = scan_fast5_data(empty_file.c_str(), &num_sequences);
		
		// REQUIRE( result == FAST5_FILE_UNREADABLE );
		
		// std::cerr << std::endl;
	// }

	SECTION("Wrong file"){
		
		std::cerr << "------TEST 23------" << std::endl;
		std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fast5";
		size_t num_sequences;
		
		int result = scan_fast5_data(wrong_file.c_str(), &num_sequences);
		
		REQUIRE( result == FAST5_FILE_UNREADABLE );
		
		std::cerr << std::endl;
	}
	
}

//TODO Get small bulk5 file to run this test
// TEST_CASE( " Get Coord Value " ) {
	
// }

//TODO Get small bulk5 file to run this test
// TEST_CASE( " Scan Bulk5 Data " ) {
	
// }

TEST_CASE( " Scan FastA Data " ) {
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 24------" << std::endl;
		std::string good_file = current_working_dir + "/good_files/fasta/fna/good_file.fna";
		size_t num_sequences;
		
		int result = scan_fasta_data(good_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 6 );
		
		std::cerr << std::endl;
	}

	SECTION("Bad file"){
		
		std::cerr << "------TEST 25------" << std::endl;
		std::string bad_file = current_working_dir + "/nope/not/real.fna";
		size_t num_sequences;
		
		int result = scan_fasta_data(bad_file.c_str(), &num_sequences);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}

	SECTION("Empty file"){
		
		std::cerr << "------TEST 26------" << std::endl;
		std::string empty_file = current_working_dir + "/empty_files/empty_test.fna";
		size_t num_sequences;
		
		int result = scan_fasta_data(empty_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 0 );
		
		std::cerr << std::endl;
	}

	// TODO: Not sure how this should really run if some of the file contents are incorrect
	SECTION("Wrong file"){
		
		std::cerr << "------TEST 27------" << std::endl;
		std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fna";
		size_t num_sequences;
		
		int result = scan_fasta_data(wrong_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 3 );
		
		std::cerr << std::endl;
	}
	
}

// TODO: get tsv file to run this test
TEST_CASE( " Scan TSV Data " ) {
	
	SECTION("Good file"){
		
		std::cerr << "------TEST 28------" << std::endl;
		std::string good_file = current_working_dir + "/good_files/good_file.tsv";
		size_t num_sequences;
		
		int result = scan_tsv_data(good_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 2 );
		
		std::cerr << std::endl;
	}

	SECTION("Bad file"){
		
		std::cerr << "------TEST 29------" << std::endl;
		std::string bad_file = current_working_dir + "/nope/not/real.tsv";
		size_t num_sequences;
		
		int result = scan_tsv_data(bad_file.c_str(), &num_sequences);
		
		REQUIRE( result == 0 );
		
		std::cerr << std::endl;
	}
	
	SECTION("Empty file"){
		
		std::cerr << "------TEST 30------" << std::endl;
		std::string empty_file = current_working_dir + "/empty_files/empty_file.tsv";
		size_t num_sequences;
		
		int result = scan_tsv_data(empty_file.c_str(), &num_sequences);
		
		REQUIRE( result == 1 );
		REQUIRE( num_sequences == 0 );
		
		std::cerr << std::endl;
	}
	
}