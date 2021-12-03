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

// TEST_CASE( " File Read " ) {
	
// }

// TEST_CASE( " Count Lines " ) {
	
// }

// TEST_CASE( " Short to Template " ) {
	
// }

// TEST_CASE( " Has Ending " ) {
	
// }

// TEST_CASE( " Is Bulk5 " ) {
	
// }

// TEST_CASE( " Check Ending " ) {
	
// }

// TEST_CASE( " Get All Files From Directory " ) {
	
// }

TEST_CASE( " Populate Tree " ) {

	std::string good_file = current_working_dir + "/good_files/good.bed";
	std::string bad_file = current_working_dir + "/nope/not/real.bed";
	std::string empty_file = current_working_dir + "/empty_files/empty.bed";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong.bed";

	ITree::interval_vector bed_intervals;

	SECTION("Good file"){

		std::cerr << "------TEST 29------" << std::endl;
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

		std::cerr << "------TEST 30------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(bad_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 31------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(empty_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 32------" << std::endl;
		int result = populateITree(bed_intervals, stringToChar(wrong_file), 1);

		REQUIRE( result == 0 );
		REQUIRE( bed_intervals.empty() == true );

		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Scan Fast5 Data " ) {
	
}

TEST_CASE( " Get Coord Value " ) {
	
}

TEST_CASE( " Scan Bulk5 Data " ) {
	
}

TEST_CASE( " Scan FastA Data " ) {
	
}

TEST_CASE( " Scan TSV Data " ) {
	
}