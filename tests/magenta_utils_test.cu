#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../magenta_utils.h"

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

TEST_CASE( " Load Index " ) {

	std::string good_file = current_working_dir + "/good_files/fasta/fasta_test";
	std::string bad_file("C://Not//a/real/path/fasta_test");
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fna";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fna";

	std::vector< std::pair<size_t, char *> > subject_offsets;
	int single_strand;

	SECTION ("Good file name") {
		
		std::cerr << "------TEST 1------" << std::endl;
		int result = load_subject_index(good_file.c_str(), subject_offsets, &single_strand);
	
		std::string name1 = "NC_000913.3";
		std::string name2 = "NC_001416";
	
		REQUIRE( result == 1 );
		REQUIRE( single_strand == 0 );
		REQUIRE( subject_offsets.size() == 2 );
	
		REQUIRE( subject_offsets.at(0).first == 9283304 );
		REQUIRE( subject_offsets.at(1).first == 9380388 );
	
		std::string off1(subject_offsets.at(0).second);
		std::string off2(subject_offsets.at(1).second);
		
		REQUIRE( off1 == name1 );
		REQUIRE( off2 == name2 );

		std::cerr << std::endl;
	}

	SECTION ("Bad file name") {

		std::cerr << "------TEST 2------" << std::endl;
		int result = load_subject_index(bad_file.c_str(), subject_offsets, &single_strand);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Empty file") {

		std::cerr << "------TEST 3------" << std::endl;
		int result = load_subject_index(empty_file.c_str(), subject_offsets, &single_strand);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Wrong file"){

		std::cerr << "------TEST 4------" << std::endl;
		int result = load_subject_index(wrong_file.c_str(), subject_offsets, &single_strand);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " Sequence Stats " ) {

	long long longest_seq_seqsize, total_seq_size;

	std::string file_good = current_working_dir + "/../sample_data/E_coli/ecoli_genome.fna";
	std::string file_bad = "C:/Not/a/real/path/ecoli_genome.fna";
	std::string file_empty = current_working_dir + "/empty_files/empty_test.fna";
	std::string file_wrong = current_working_dir + "/wrong_files/wrong_test.fna";

	SECTION ("Good file name"){

		std::cerr << "------TEST 5------" << std::endl;	
		int result = getSequenceStats(stringToChar(file_good), &longest_seq_seqsize, &total_seq_size, false);
	
		REQUIRE( result == 1 );
		REQUIRE( longest_seq_seqsize == 4641652);
		REQUIRE( total_seq_size == 4690154);

		std::cerr << std::endl;
	}

	SECTION ("Bad file name"){

		std::cerr << "------TEST 6------" << std::endl;
		int result = getSequenceStats(stringToChar(file_bad), &longest_seq_seqsize, &total_seq_size, false);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Empty file") {

		std::cerr << "------TEST 7------" << std::endl;
		int result = getSequenceStats(stringToChar(file_empty), &longest_seq_seqsize, &total_seq_size, false);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Wrong file"){

		std::cerr << "------TEST 8------" << std::endl;
		int result = getSequenceStats(stringToChar(file_wrong), &longest_seq_seqsize, &total_seq_size, false);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " Get All Fast5 Files " ) {

	std::vector<std::string> all_fast5_files;

	std::string name1;
	std::string name2;
	std::string name3;
	std::string name4;
	std::string name5;

	if(ONWINDOWS){
		name1 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
		name2 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
		name3 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_493_ch_223_strand.fast5";
		name4 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_519_ch_85_strand.fast5";
		name5 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_664_ch_195_strand.fast5";
	} else{
		name1 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
		name2 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_519_ch_85_strand.fast5";
		name3 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_493_ch_223_strand.fast5";
		name4 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
		name5 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_664_ch_195_strand.fast5";
	}

	std::string dir_path_good = current_working_dir + "/good_files/fast5/";
	std::string dir_path_bad = "C:/Not/a/real/path/fast5/";
	std::string dir_path_empty = current_working_dir + "/empty_files/empty/";
	std::string dir_path_good_bs = current_working_dir + "/good_files/fast5";
	
	SECTION ("Good file path"){

		std::cerr << "------TEST 9------" << std::endl;		
		int result = getAllFast5Files(stringToChar(dir_path_good), all_fast5_files);
	
		REQUIRE( result == 1 );
		REQUIRE( all_fast5_files.size() == 5 );
	
		REQUIRE( all_fast5_files.at(0) == name1 );
		REQUIRE( all_fast5_files.at(1) == name2 );
		REQUIRE( all_fast5_files.at(2) == name3 );
		REQUIRE( all_fast5_files.at(3) == name4 );
		REQUIRE( all_fast5_files.at(4) == name5 );

		std::cerr << std::endl;
	}

	SECTION ("Bad file path"){

		std::cerr << "------TEST 10------" << std::endl;
		int result = getAllFast5Files(stringToChar(dir_path_bad), all_fast5_files);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Empty directory"){

		std::cerr << "------TEST 11------" << std::endl;
		int result = getAllFast5Files(stringToChar(dir_path_empty), all_fast5_files);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	
	SECTION ("Good file path no backslash"){

		std::cerr << "------TEST 12------" << std::endl;
		int result = getAllFast5Files(stringToChar(dir_path_good_bs), all_fast5_files);
	
		REQUIRE( result == 1 );
		REQUIRE( all_fast5_files.size() == 5 );
	
		REQUIRE( all_fast5_files.at(0) == name1 );
		REQUIRE( all_fast5_files.at(1) == name2 );
		REQUIRE( all_fast5_files.at(2) == name3 );
		REQUIRE( all_fast5_files.at(3) == name4 );
		REQUIRE( all_fast5_files.at(4) == name5 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " Fast5 Stats " ) {

	std::string name1 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
	std::string name2 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_352_ch_156_strand.fast5";
	std::string name3 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_493_ch_223_strand.fast5";
	std::string name4 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_519_ch_85_strand.fast5";
	std::string name5 = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_664_ch_195_strand.fast5";
	
	std::vector<std::string> good_fast5_files;
	good_fast5_files.push_back(name1);
	good_fast5_files.push_back(name2);
	good_fast5_files.push_back(name3);
	good_fast5_files.push_back(name4);
	good_fast5_files.push_back(name5);

	long long longest_seq_seqsize, total_seq_size;
	
	SECTION ("Good files using raw"){
		
		std::cerr << "------TEST 13------" << std::endl;
		int result = getFast5Stats(good_fast5_files, &longest_seq_seqsize, &total_seq_size, 1, 0);

		REQUIRE( result == 1 );

		REQUIRE( longest_seq_seqsize == 92662 );
		REQUIRE( total_seq_size == 287188 );

		std::cerr << std::endl;
	}

	SECTION ("Good files using events"){

		std::cerr << "------TEST 14------" << std::endl;
		int result = getFast5Stats(good_fast5_files, &longest_seq_seqsize, &total_seq_size, 0, 0);
	
		REQUIRE( result == 1 );
	
		REQUIRE( longest_seq_seqsize == 5313 );
		REQUIRE( total_seq_size == 16555 );

		std::cerr << std::endl;
	}

	SECTION ("One bad file path"){

		std::cerr << "------TEST 15------" << std::endl;
		std::string bad_name = "C:/Not/a/real/path/fast5/";
		
		good_fast5_files.push_back(bad_name);

		int result = getFast5Stats(good_fast5_files, &longest_seq_seqsize, &total_seq_size, 0, 0);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("No file paths"){

		std::cerr << "------TEST 16------" << std::endl;
		std::vector<std::string> empty_fast5_files;

		int result = getFast5Stats(empty_fast5_files, &longest_seq_seqsize, &total_seq_size, 0, 0);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION ("Wrong file"){

		std::cerr << "------TEST 17------" << std::endl;
		std::string wrong_name1 = current_working_dir + "/wrong_files/fast5/oops.fast5";
		std::string wrong_name2 = current_working_dir + "/wrong_files/fast5/ohno.fast5";
		std::string wrong_name3 = current_working_dir + "/wrong_files/fast5/test.txt";

		std::vector<std::string> wrong_fast5_files;
		// wrong_fast5_files.push_back(wrong_name1);
		wrong_fast5_files.push_back(wrong_name2);
		wrong_fast5_files.push_back(wrong_name3);

		int result = getFast5Stats(wrong_fast5_files, &longest_seq_seqsize, &total_seq_size, 0, 0);
	
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Get Sequence " ) {

	std::string file_good = current_working_dir + "/../sample_data/E_coli/ecoli_genome.fna";
	long long longest_seq_seqsize = 4641652;
	long long total_seq_size = 4690154;

	long subject_bases_length = longest_seq_seqsize + 1;
	char * subject_bases = (char *) calloc(subject_bases_length, sizeof(char)); // +1 to allow null terminator
	if(NULL == subject_bases){
		std::cerr << "Could not allocate memory for subject bases (" << (subject_bases_length*sizeof(char)) << " bytes), aborting" << std::endl;
		exit(1);
	}

	long long subject_length = 0;
	std::string subject_name;
	std::string buffered_header;
	int dots_printed = -1; // tell fasta2pAs not to print dots
	long bytes_per_dot = total_seq_size/100;
	

	SECTION("Good file"){

		std::cerr << "------TEST 18------" << std::endl;
		std::ifstream sfile(stringToChar(file_good), std::ios::binary);

		int result = getSequence(sfile, subject_bases, &subject_length, subject_name, buffered_header, &dots_printed, bytes_per_dot);

		REQUIRE( result == 1 );

		REQUIRE( subject_length == 4641652);
		REQUIRE( subject_bases[0] == 'A' );
		REQUIRE( subject_bases[1] == 'G' );
		REQUIRE( subject_bases[2] == 'C' );
		REQUIRE( subject_bases[3] == 'T' );

		sfile.close();

		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 19------" << std::endl;
		std::string file_empty = current_working_dir + "/empty_files/empty_test.fna";
		std::ifstream sfile(stringToChar(file_empty), std::ios::binary);

		int result = getSequence(sfile, subject_bases, &subject_length, subject_name, buffered_header, &dots_printed, bytes_per_dot);

		REQUIRE( result == 0 );
		sfile.close();

		std::cerr << std::endl;

	}

	SECTION("Empty file with FastA record name"){

		std::cerr << "------TEST 20------" << std::endl;
		std::string file_empty = current_working_dir + "/empty_files/empty_ish_test.fna";
		std::ifstream sfile(stringToChar(file_empty), std::ios::binary);

		int result = getSequence(sfile, subject_bases, &subject_length, subject_name, buffered_header, &dots_printed, bytes_per_dot);

		REQUIRE( result == 0 );
		sfile.close();

		std::cerr << std::endl;

	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 21------" << std::endl;
		std::string file_wrong = current_working_dir + "/wrong_files/wrong_test.fna";
		std::ifstream sfile(stringToChar(file_wrong), std::ios::binary);

		int result = getSequence(sfile, subject_bases, &subject_length, subject_name, buffered_header, &dots_printed, bytes_per_dot);

		REQUIRE( result == 0 );
		sfile.close();

		std::cerr << std::endl;
	}
}

TEST_CASE( " Sample Rate " ) {

	std::string file_fast5_good = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
	std::string file_fast5_bad = current_working_dir + "/not/a/path.fast5";
	std::string file_fast5_empty = current_working_dir + "/empty_files/empty_test.fast5";
	std::string file_fast5_wrong = current_working_dir + "/wrong_files/wrong_test.fast5";

	int bulk = 1;
	int not_bulk = 0;

	// std::string file_bulk5_good = current_working_dir + 

	SECTION("Good Fast5 file not bulk"){

		std::cerr << "------TEST 22------" << std::endl;
		double sample_rate = getSampleRate(stringToChar(file_fast5_good), not_bulk);

		REQUIRE( sample_rate == 4000 );

		std::cerr << std::endl;
	}

	SECTION("Good Bulk5 file bulk"){

		std::cerr << "------TEST 23------" << std::endl;
		std::cerr << "Not implemented" << std::endl;
		std::cerr << std::endl;

	}

	SECTION("Bad file"){

		std::cerr << "------TEST 24------" << std::endl;
		double sample_rate = getSampleRate(stringToChar(file_fast5_bad), not_bulk);

		REQUIRE( sample_rate == 0 );

		std::cerr << std::endl;

	}

	SECTION("Empty file"){

		std::cerr << "------TEST 25------" << std::endl;
		double sample_rate = getSampleRate(stringToChar(file_fast5_empty), not_bulk);

		REQUIRE( sample_rate == 0 );

		std::cerr << std::endl;

	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 26------" << std::endl;
		double sample_rate = getSampleRate(stringToChar(file_fast5_wrong), not_bulk);

		REQUIRE( sample_rate == 0 );

		std::cerr << std::endl;	

	}

	SECTION("Good Fast5 file bulk"){

		std::cerr << "------TEST 27------" << std::endl;
		double sample_rate = getSampleRate(stringToChar(file_fast5_good), bulk);

		REQUIRE( sample_rate == 0 );

		std::cerr << std::endl;

	}

	SECTION("Good Bulk5 file not bulk"){

		std::cerr << "------TEST 28------" << std::endl;
		std::cerr << "Not implemented" << std::endl;
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

// TEST_CASE( " Populate Subject - FastA " ) {

	// std::string good_file = current_working_dir + "/../sample_data/E_coli/ecoli_genome.fna";
	// std::string bad_file = "not/real/path.fna";
	// std::string empty_file = current_working_dir + "/empty_files/empty_test.fna";
	// std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fna";

	// short* subject_values;
	// unsigned long long int num_subject_values;
	// std::vector< std::pair<size_t, char *> > subject_offsets;

	// int sig_type_mean = MEAN_SIGNAL;
	// int sig_type_std = STDDEV_SIGNAL;

	// int rna_no = 0;
	// int rna_yes = 1;

	// int complement_no = 0;
	// int complement_yes = 1;

	// int complement_only_no = 0;
	// int complement_only_yes = 1;

	// SECTION("Good file"){

		// std::cerr << "------TEST 33------" << std::endl;
		// int result = populateSubjectWithFastA(&subject_values, &num_subject_values, stringToChar(good_file),
										  // subject_offsets, sig_type_mean, rna_no, complement_no, complement_only_no, 1);

		// REQUIRE( result == 1 );
		// REQUIRE( num_subject_values == 4690154 );

		// // REQUIRE( subject_values[0] == 'A' );		// TODO: Ask about what these values should be after they're read in
		// // REQUIRE( subject_values[1] == 'G' );
		// // REQUIRE( subject_values[2] == 'C' );
		// // REQUIRE( subject_values[3] == 'T' );
		// std::cerr << std::endl;

	// }

	// SECTION("Bad file"){

		// std::cerr << "------TEST 34------" << std::endl;
		// int result = populateSubjectWithFastA(&subject_values, &num_subject_values, stringToChar(bad_file),
										  // subject_offsets, sig_type_mean, rna_no, complement_no, complement_only_no, 1);

		// REQUIRE( result == 0 );

		// std::cerr << std::endl;

	// }

	// SECTION("Empty file"){

		// std::cerr << "------TEST 35------" << std::endl;
		// int result = populateSubjectWithFastA(&subject_values, &num_subject_values, stringToChar(empty_file),
										  // subject_offsets, sig_type_mean, rna_no, complement_no, complement_only_no, 1);

		// REQUIRE( result == 0 );

		// std::cerr << std::endl;

	// }

	// SECTION("Wrong file"){

		// std::cerr << "------TEST 36------" << std::endl;
		// int result = populateSubjectWithFastA(&subject_values, &num_subject_values, stringToChar(wrong_file),
										  // subject_offsets, sig_type_mean, rna_no, complement_no, complement_only_no, 1);

		// REQUIRE( result == 0 );

		// std::cerr << std::endl;

	// }
// }

TEST_CASE( " Populate Subject Buffer - Fast5 " ) {

	std::string good_file = current_working_dir + "/good_files/fast5/";
	std::string bad_file = "not/real/";
	std::string empty_file = current_working_dir + "/empty_files/empty/";
	std::string wrong_file = current_working_dir + "/wrong_files/fast5/";

	short* subject_values;
	unsigned long long int num_subject_values;
	std::vector< std::pair<size_t, char *> > subject_offsets;

	SECTION("Good file"){

		std::cerr << "------TEST 37------" << std::endl;
		int result = populateSubjectBuffFast5(&subject_values, &num_subject_values, stringToChar(good_file), subject_offsets, 1);

		REQUIRE( result == 1 );
		
		REQUIRE( num_subject_values == 287188 );
		if(ONWINDOWS){
			REQUIRE( subject_values[0] == 774 );
			REQUIRE( subject_values[84847] == 703 );
			REQUIRE( subject_values[133581] == 1043 );
			REQUIRE( subject_values[226243] == 882 );
			REQUIRE( subject_values[280032] == 665 );
		} else {
			REQUIRE( subject_values[0] == 774 );
			REQUIRE( subject_values[84847] == 882 );
			REQUIRE( subject_values[138636] == 1043 );
			REQUIRE( subject_values[231298] == 703 );
			REQUIRE( subject_values[280032] == 665 );
		}

		std::cerr << std::endl;
	}

	SECTION("Bad file"){

		std::cerr << "------TEST 38------" << std::endl;
		int result = populateSubjectBuffFast5(&subject_values, &num_subject_values, stringToChar(bad_file), subject_offsets, 1);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 39------" << std::endl;
		int result = populateSubjectBuffFast5(&subject_values, &num_subject_values, stringToChar(empty_file), subject_offsets, 1);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 40------" << std::endl;
		int result = populateSubjectBuffFast5(&subject_values, &num_subject_values, stringToChar(wrong_file), subject_offsets, 1);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
	
}

TEST_CASE( " Sample Name " ) {

	std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
	std::string bad_file = "not/real/path.fast5";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fast5";

	char* sample_name;

	SECTION("Good file"){

		std::cerr << "------TEST 41------" << std::endl;
		int result = getSampleName(stringToChar(good_file), &sample_name);

		std::string sample_string_name(sample_name);
		std::string name = "msi_20180314_fah40082_mn24809_sequencing_run_mt237_3_brdu_mcginty_54076_Read_229_222";

		REQUIRE( result == 1 );
		REQUIRE( sample_string_name == name );

		std::cerr << std::endl;
	}

	SECTION("Bad file"){

		std::cerr << "------TEST 42------" << std::endl;
		int result = getSampleName(stringToChar(bad_file), &sample_name);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 43------" << std::endl;
		int result = getSampleName(stringToChar(empty_file), &sample_name);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 44------" << std::endl;
		int result = getSampleName(stringToChar(wrong_file), &sample_name);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}
}

TEST_CASE( " Read Fast5 " ) {

	std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
	std::string bad_file = "not/real/path.fast5";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fast5";

	std::string name = "msi_20180314_fah40082_mn24809_sequencing_run_mt237_3_brdu_mcginty_54076_Read_229_222";

	long long num_events;
	char* sample_name;
	double sample_rate;
	int use_raw = 1;
	int not_raw = 0;

	short* short_events;
	float* float_events;

	SECTION("Good file using raw"){

		std::cerr << "------TEST 45------" << std::endl;
		short_events = readEventsFromFast5<short>(stringToChar(good_file), &num_events, &sample_name, &sample_rate, use_raw, 1);

		REQUIRE( short_events != 0 );

		std::string sample_string_name(sample_name);
		REQUIRE( num_events == 7156 );
		REQUIRE( sample_string_name == name );
		REQUIRE( sample_rate == 4000 );
		REQUIRE( short_events[0] == 665 );

		std::cerr << std::endl;
	}

	SECTION("Bad file"){

		std::cerr << "------TEST 46------" << std::endl;
		short_events = readEventsFromFast5<short>(stringToChar(bad_file), &num_events, &sample_name, &sample_rate, use_raw, 1);

		REQUIRE( short_events == 0 );
		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 47------" << std::endl;
		short_events = readEventsFromFast5<short>(stringToChar(empty_file), &num_events, &sample_name, &sample_rate, use_raw, 1);

		REQUIRE( short_events == 0 );
		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 48------" << std::endl;
		short_events = readEventsFromFast5<short>(stringToChar(wrong_file), &num_events, &sample_name, &sample_rate, use_raw, 1);

		REQUIRE( short_events == 0 );
		std::cerr << std::endl;
	}

	SECTION("Good file using non-raw"){

		std::cerr << "------TEST 49------" << std::endl;
		float_events = readEventsFromFast5<float>(stringToChar(good_file), &num_events, &sample_name, &sample_rate, not_raw, 1);

		REQUIRE( float_events != 0 );

		std::string sample_string_name(sample_name);
		REQUIRE( num_events == 375 );
		REQUIRE( sample_string_name == name );
		REQUIRE( sample_rate == 4000 );
		REQUIRE( float_events[0] == 115.72782f );

		std::cerr << std::endl;
	}
}

TEST_CASE( "Populate Subject - Fast5 " ) {

	std::string good_file = current_working_dir + "/good_files/fast5/MSI_20180314_FAH40082_MN24809_sequencing_run_mt237_3_brdu_mcginty_54076_read_229_ch_222_strand.fast5";
	std::string bad_file = "not/real/path.fast5";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fast5";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fast5";

	std::string name = "msi_20180314_fah40082_mn24809_sequencing_run_mt237_3_brdu_mcginty_54076_Read_229_222";

	short* short_subject = (short *) calloc(7156, sizeof(short));
	float* float_subject = (float *) calloc(375, sizeof(float));

	long long subject_length;
	std::string subject_name;
	int dots_printed = 0;
	long bytes_per_dot = 7156/102;

	int use_raw = 1;
	int not_raw = 0;

	SECTION("Good file using raw"){

		std::cerr << "------TEST 50------" << std::endl;
		int result = populateSubjectWithFast5<short>(good_file, short_subject, &subject_length, subject_name, &dots_printed, bytes_per_dot, use_raw, 1);

		REQUIRE( result == 1 );

		std::string sample_string_name(subject_name);
		REQUIRE( subject_length == 7156 );
		REQUIRE( sample_string_name == name );
		REQUIRE( short_subject[0] == 665 );

		std::cerr << std::endl;
	}

	SECTION("Bad file"){

		std::cerr << "------TEST 51------" << std::endl;
		int result = populateSubjectWithFast5<short>(bad_file, short_subject, &subject_length, subject_name, &dots_printed, bytes_per_dot, use_raw, 1);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}

	SECTION("Empty file"){

		std::cerr << "------TEST 52------" << std::endl;
		int result = populateSubjectWithFast5<short>(empty_file, short_subject, &subject_length, subject_name, &dots_printed, bytes_per_dot, use_raw, 1);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}

	SECTION("Wrong file"){

		std::cerr << "------TEST 53------" << std::endl;
		int result = populateSubjectWithFast5<short>(wrong_file, short_subject, &subject_length, subject_name, &dots_printed, bytes_per_dot, use_raw, 1);

		REQUIRE( result == 0 );
		std::cerr << std::endl;
	}

	SECTION("Good file using non-raw"){

		std::cerr << "------TEST 54------" << std::endl;
		int result = populateSubjectWithFast5<float>(good_file, float_subject, &subject_length, subject_name, &dots_printed, bytes_per_dot, not_raw, 1);

		REQUIRE( result == 1 );

		std::string sample_string_name(subject_name);
		REQUIRE( subject_length == 375 );
		REQUIRE( sample_string_name == name );
		REQUIRE( float_subject[0] == 115.72782f );

		std::cerr << std::endl;
	}

	free(short_subject);
	free(float_subject);
}

TEST_CASE( " FastA2Prefixes " ) {

	std::string good_file = current_working_dir + "/good_files/fasta/fna/prefix_test.fna";
	std::string bad_file = current_working_dir + "/not/a/real/file.fna";
	std::string empty_file = current_working_dir + "/empty_files/empty_test.fna";
	std::string wrong_file = current_working_dir + "/wrong_files/wrong_test.fna";

	std::vector<std::string> file_contents;
	std::ifstream resultfile;

	int reverse = 1;
	int not_reverse = 0;
	int minimal = 1;
	int not_minimal = 0;

	int requested_suffix_length_three = 3;

	int verbose = 1;

	SECTION("Good file"){
		std::cerr << "------TEST 55------" << std::endl;
		int result = fasta2prefixes(stringToChar(good_file), file_contents, not_reverse, requested_suffix_length_three, not_minimal, verbose);

		REQUIRE( result == 1 );

		REQUIRE( file_contents.size() == 12 );

		REQUIRE( file_contents.at(0) == ">seq1 kept_len=21 prefix_len=18 suffix_len=4" );
		REQUIRE( file_contents.at(1) == "AAGCTTTTGCCCAGTCAAGCT" );
		REQUIRE( file_contents.at(2) == ">seq2 kept_len=21 prefix_len=18 suffix_len=4" );
		REQUIRE( file_contents.at(3) == "AAGCTTTTGCCCAGTCAAAAA" );
		REQUIRE( file_contents.at(4) == ">seq3 kept_len=20 prefix_len=17 suffix_len=5" );
		REQUIRE( file_contents.at(5) == "AAGCTTTTGCCCAGTCATAA" );
		REQUIRE( file_contents.at(6) == ">seq4 kept_len=8 prefix_len=5 suffix_len=11" );
		REQUIRE( file_contents.at(7) == "AAGCTATT" );
		REQUIRE( file_contents.at(8) == ">seq5 kept_len=4 prefix_len=1 suffix_len=10" );
		REQUIRE( file_contents.at(9) == "GCTT" );
		REQUIRE( file_contents.at(10) == ">seq6 kept_len=4 prefix_len=1 suffix_len=10" );
		REQUIRE( file_contents.at(11) == "GTTT" );

		std::cerr << std::endl;
	}

	SECTION("Bad file"){
		std::cerr << "------TEST 56------" << std::endl;
		int result = fasta2prefixes(stringToChar(bad_file), file_contents, reverse, requested_suffix_length_three, minimal, verbose);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Empty file"){
		std::cerr << "------TEST 57------" << std::endl;
		int result = fasta2prefixes(stringToChar(empty_file), file_contents, reverse, requested_suffix_length_three, minimal, verbose);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Wrong file"){
		std::cerr << "------TEST 58------" << std::endl;
		int result = fasta2prefixes(stringToChar(wrong_file), file_contents, reverse, requested_suffix_length_three, minimal, verbose);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " Get Prefix Sequence Stats " ) {

	std::vector<std::string> good_prefixes;
	good_prefixes.push_back(">seq1 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAGCT");
	good_prefixes.push_back(">seq2 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAAAA");
	good_prefixes.push_back(">seq3 kept_len=20 prefix_len=17 suffix_len=5");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCATAA" );
	good_prefixes.push_back(">seq4 kept_len=8 prefix_len=5 suffix_len=11");
	good_prefixes.push_back("AAGCTATT");
	good_prefixes.push_back(">seq5 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GCTT");
	good_prefixes.push_back(">seq6 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GTTT");

	std::vector<std::string> empty_prefixes;

	std::vector<std::string> wrong_prefixes;
	wrong_prefixes.push_back("o");
	wrong_prefixes.push_back("124");
	wrong_prefixes.push_back("");
	wrong_prefixes.push_back("");

	long long longest_seq_seqsize = 0;
	long long total_seq_size = 0;
	bool print_pct_dots = false;

	SECTION("Good prefixes"){
		std::cerr << "------TEST 59------" << std::endl;
		int result = getSequenceStatsPrefix(good_prefixes, &longest_seq_seqsize, &total_seq_size, print_pct_dots);

		REQUIRE( result == 1 );

		REQUIRE( longest_seq_seqsize == 21 );
		REQUIRE( total_seq_size == 78 );

		std::cerr << std::endl;
	}


	SECTION("Empty prefixes"){
		std::cerr << "------TEST 60------" << std::endl;
		int result = getSequenceStatsPrefix(empty_prefixes, &longest_seq_seqsize, &total_seq_size, print_pct_dots);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Wrong prefixes"){
		std::cerr << "------TEST 61------" << std::endl;
		int result = getSequenceStatsPrefix(wrong_prefixes, &longest_seq_seqsize, &total_seq_size, print_pct_dots);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

}

TEST_CASE ( "Get Sequence Prefix" ){

	std::vector<std::string> good_prefixes;
	good_prefixes.push_back(">seq1 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAGCT");
	good_prefixes.push_back(">seq2 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAAAA");
	good_prefixes.push_back(">seq3 kept_len=20 prefix_len=17 suffix_len=5");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCATAA" );
	good_prefixes.push_back(">seq4 kept_len=8 prefix_len=5 suffix_len=11");
	good_prefixes.push_back("AAGCTATT");
	good_prefixes.push_back(">seq5 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GCTT");
	good_prefixes.push_back(">seq6 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GTTT");

	std::vector<std::string> wrong_prefixes;
	wrong_prefixes.push_back("o");
	wrong_prefixes.push_back("123");

	int subject_bases_length = 22;
	char *subject_bases_buffer  = (char *) calloc(subject_bases_length, sizeof(char));
	long total_seqsize = 78;

	long long num_subject_bases = 0;
	std::string subject_name;
	std::string buffered_header;
	int num_dots_printed = -1;

	int result;

	SECTION("Good prefix"){
		// Pretend this is in a for loop
		std::cerr << "------TEST 62------" << std::endl;
		int count = 0;

		result =  getSequencePrefix(good_prefixes, &count, subject_bases_buffer, &num_subject_bases, subject_name, buffered_header, &num_dots_printed, total_seqsize);
		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq2 kept_len=21 prefix_len=18 suffix_len=4" );
		REQUIRE( subject_name == "seq1" );
		REQUIRE( num_subject_bases == 21 );
		REQUIRE( subject_bases_buffer[0] == 'A' );
		count++;

		result =  getSequencePrefix(good_prefixes, &count, subject_bases_buffer, &num_subject_bases, subject_name, buffered_header, &num_dots_printed, total_seqsize);
		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq3 kept_len=20 prefix_len=17 suffix_len=5" );
		REQUIRE( subject_name == "seq2" );
		REQUIRE( num_subject_bases == 21 );
		REQUIRE( subject_bases_buffer[0] == 'A' );
		count++;

		result =  getSequencePrefix(good_prefixes, &count, subject_bases_buffer, &num_subject_bases, subject_name, buffered_header, &num_dots_printed, total_seqsize);
		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq4 kept_len=8 prefix_len=5 suffix_len=11" );
		REQUIRE( subject_name == "seq3" );
		REQUIRE( num_subject_bases == 20 );
		REQUIRE( subject_bases_buffer[0] == 'A' );

		std::cerr << std::endl;
	}

	SECTION("Wrong prefix"){
		std::cerr << "------TEST 63------" << std::endl;
		int count = 0;
		result =  getSequencePrefix(wrong_prefixes, &count, subject_bases_buffer, &num_subject_bases, subject_name, buffered_header, &num_dots_printed, total_seqsize);
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( " FastA Prefix to pAs " ){

	std::vector<std::string> good_prefixes;
	good_prefixes.push_back(">seq1 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAGCT");
	good_prefixes.push_back(">seq2 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAAAA");
	good_prefixes.push_back(">seq3 kept_len=20 prefix_len=17 suffix_len=5");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCATAA" );
	good_prefixes.push_back(">seq4 kept_len=8 prefix_len=5 suffix_len=11");
	good_prefixes.push_back("AAGCTATT");
	good_prefixes.push_back(">seq5 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GCTT");
	good_prefixes.push_back(">seq6 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GTTT");

	std::vector<std::string> wrong_prefixes;
	wrong_prefixes.push_back("o");
	wrong_prefixes.push_back("123");

	int subject_bases_length = 22;
	char *subject_bases_buffer  = (char *) calloc(subject_bases_length, sizeof(char));
	long total_seqsize = 78;

	long long num_subject_bases = 0;
	std::string subject_name;
	std::string buffered_header;
	int num_dots_printed = -1;
	long signal_length = 0;

	int sig_type_mean = MEAN_SIGNAL;
	int sig_type_std = STDDEV_SIGNAL;

	int complement = 1;
	int no_complement = 0;

	int complement_only = 1;
	int no_complement_only = 0;

	int rna = 1;
	int no_rna = 0;

	int verbose = 1;
	int no_verbose = 0;

	int result;

	short* subject_pAs_buffer = NULL;

	SECTION("Good prefix"){
		std::cerr << "------TEST 64------" << std::endl;
		int count = 0;
		// Pretend this is a big loop
		result = fastaPrefix2pAs(good_prefixes, &count, subject_bases_buffer, &subject_pAs_buffer, &num_subject_bases, subject_name,
								buffered_header, &num_dots_printed, total_seqsize, &signal_length, sig_type_std,
								no_complement, no_complement_only, rna, verbose);

		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq2 kept_len=21 prefix_len=18 suffix_len=4" );
		REQUIRE( subject_name == "seq1" );
		REQUIRE( num_subject_bases == 21 );
		REQUIRE( subject_bases_buffer[0] == 'A' );
		count++;

		result = fastaPrefix2pAs(good_prefixes, &count, subject_bases_buffer, &subject_pAs_buffer, &num_subject_bases, subject_name,
								buffered_header, &num_dots_printed, total_seqsize, &signal_length, sig_type_std,
								no_complement, no_complement_only, rna, verbose);
		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq3 kept_len=20 prefix_len=17 suffix_len=5" );
		REQUIRE( subject_name == "seq2" );
		REQUIRE( num_subject_bases == 21 );
		count++;

		result = fastaPrefix2pAs(good_prefixes, &count, subject_bases_buffer, &subject_pAs_buffer, &num_subject_bases, subject_name,
								buffered_header, &num_dots_printed, total_seqsize, &signal_length, sig_type_std,
								no_complement, no_complement_only, rna, verbose);
		REQUIRE( result == 1 );
		REQUIRE( buffered_header == ">seq4 kept_len=8 prefix_len=5 suffix_len=11" );
		REQUIRE( subject_name == "seq3" );
		REQUIRE( num_subject_bases == 20 );
		REQUIRE( subject_bases_buffer[0] == 'A' );
		count++;

		std::cerr << std::endl;
	}

	SECTION("Wrong prefix"){
		std::cerr << "------TEST 65------" << std::endl;
		int count = 0;
		result = fastaPrefix2pAs(wrong_prefixes, &count, subject_bases_buffer, &subject_pAs_buffer, &num_subject_bases, subject_name,
								buffered_header, &num_dots_printed, total_seqsize, &signal_length, sig_type_std,
								no_complement, no_complement_only, rna, verbose);
		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}

TEST_CASE( "Pop Subject With FastA Prefix" ){

	std::vector<std::string> good_prefixes;
	good_prefixes.push_back(">seq1 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAGCT");
	good_prefixes.push_back(">seq2 kept_len=21 prefix_len=18 suffix_len=4");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCAAAAA");
	good_prefixes.push_back(">seq3 kept_len=20 prefix_len=17 suffix_len=5");
	good_prefixes.push_back("AAGCTTTTGCCCAGTCATAA" );
	good_prefixes.push_back(">seq4 kept_len=8 prefix_len=5 suffix_len=11");
	good_prefixes.push_back("AAGCTATT");
	good_prefixes.push_back(">seq5 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GCTT");
	good_prefixes.push_back(">seq6 kept_len=4 prefix_len=1 suffix_len=10");
	good_prefixes.push_back("GTTT");

	std::vector<std::string> empty_prefixes;

	std::vector<std::string> wrong_prefixes;
	wrong_prefixes.push_back("o");
	wrong_prefixes.push_back("123");

	short* subject_values;
	unsigned long long int num_subject_values = 0;

	std::vector< std::pair<size_t, char *> > subject_offsets;

	int sig_type_mean = MEAN_SIGNAL;
	int sig_type_std = STDDEV_SIGNAL;

	int complement = 1;
	int no_complement = 0;

	int complement_only = 1;
	int no_complement_only = 0;

	int rna = 1;
	int no_rna = 0;

	int verbose = 1;
	int no_verbose = 0;

	int result;

	SECTION("Good prefix"){
		std::cerr << "------TEST 66------" << std::endl;
		result = populateSubjectWithFastAPrefix(&subject_values, &num_subject_values, good_prefixes, subject_offsets,
												sig_type_std, rna, no_complement, no_complement_only, verbose);

		REQUIRE( result == 1 );
		REQUIRE( num_subject_values == 78 );

		REQUIRE( subject_offsets.size() == 6 );
		// REQUIRE( subject_offsets.at(0).first == 8 );
		REQUIRE( subject_offsets.at(0).first == 21 );
		REQUIRE( subject_offsets.at(1).first == 42 );
		REQUIRE( subject_offsets.at(2).first == 62 );
		REQUIRE( subject_offsets.at(3).first == 70 );
		REQUIRE( subject_offsets.at(4).first == 74 );
		REQUIRE( subject_offsets.at(5).first == 78 );

		// REQUIRE( std::string(subject_offsets.at(0).second) == "seq4" );
		REQUIRE( std::string(subject_offsets.at(0).second) == "seq1" );
		REQUIRE( std::string(subject_offsets.at(1).second) == "seq2" );
		REQUIRE( std::string(subject_offsets.at(2).second) == "seq3" );
		REQUIRE( std::string(subject_offsets.at(3).second) == "seq4" );
		REQUIRE( std::string(subject_offsets.at(4).second) == "seq5" );
		REQUIRE( std::string(subject_offsets.at(5).second) == "seq6" );

		std::cerr << std::endl;
	}

	SECTION("Empty prefix"){
		std::cerr << "------TEST 67------" << std::endl;
		result = populateSubjectWithFastAPrefix(&subject_values, &num_subject_values, empty_prefixes, subject_offsets,
												sig_type_std, rna, no_complement, no_complement_only, verbose);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}

	SECTION("Wrong prefix"){
		std::cerr << "------TEST 68------" << std::endl;
		result = populateSubjectWithFastAPrefix(&subject_values, &num_subject_values, wrong_prefixes, subject_offsets,
												sig_type_std, rna, no_complement, no_complement_only, verbose);

		REQUIRE( result == 0 );

		std::cerr << std::endl;
	}
}



// TEST_CASE( " Bulk5 Stats " ) {

	// long long longest_seq_seqsize, total_seq_size;
	// std::map<char*, std::vector< std::pair <long long,long long> > > channel_ranges;

	// SECTION ("Good file"){

		// std::cerr << "------TEST 18------" << std::endl;
		// std::string good_file = current_working_dir + "/good_files/bulk5/MSI_20180528_FAH82910_MN24809_sequencing_run_MT239_4_33059.fast5";

		// int result = getBulk5Stats(stringToChar(good_file), &longest_seq_seqsize, &total_seq_size, channel_ranges, false, 1);

		// REQUIRE( result == 1 );

		// std::cerr << std::endl;
	// }

// }

// TEST_CASE( " Is Bulk5 " ) {

// }

// TEST_CASE( " Populate Subject Buffer - Bulk5 " ) {

// }

// TEST_CASE( " Bulk5 Name " ) {

// }

// TEST_CASE( " Read Bulk5 " ) {

// }

// TEST_CASE( " Populate Subject - Bulk5 " ) {

// }
