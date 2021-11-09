/*
 *
 * Copyright 2015, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
 
#define QTYPE_float
#define QTYPE_ACC_float
//#define QTYPE_short
//#define QTYPE_ACC_short
//#define QTYPE_half
//#define QTYPE_ACC_half

#include "ReadUntilClient.h"
#include "flash_dtw.cuh"
#include "dtw.hpp"
#include "segmentation.hpp"
// #include "cuda_utils.h" // CUERR()

#if defined(_WIN32)
	#include <conio.h>
	#include <windows.h>
	extern "C"{
		#include "getopt.h"
	}
	#include <direct.h>
	#define GetCurrentDir _getcwd
	#define FILE_SEPARATOR "\\"
#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#define FILE_SEPARATOR "/"
#endif

#include <signal.h>

// bool hasEnding (std::string const &fullString, std::string const &ending) {
    // if (fullString.length() >= ending.length()) {
        // return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    // } else {
        // return false;
    // }
// }

std::ofstream log_file;
void handler(int s){
	printf("Caught signal %d. Closing log\n",s);
	log_file << "Caught signal " << s << std::endl;
	log_file.close();
	exit(1); 	
}

int flash_dtw_callback(QTYPE* query_values, unsigned long long int num_query_values, char* query_name, float colDev, float maxP, float maxFDR, float maxRanks, float znorm, match_record** results, float miniDtwSize, float miniDtwWarp, bool record_match_anchors, float useFastCalc, bool use_std, int use_hard_dtw){
	int num_results = 0;
	flash_dtw(query_values, num_query_values, query_name, colDev, maxP, maxFDR, maxRanks, znorm, results, &num_results, miniDtwSize, miniDtwWarp, record_match_anchors, useFastCalc, use_std, use_hard_dtw);
	return num_results;
}

template<typename T>
void adaptive_segmentation_callback(T **sequences, size_t *seq_lengths, int num_seqs, int min_segment_length, T ***segmented_sequences, size_t **segmented_seq_lengths){
	adaptive_segmentation(sequences, seq_lengths, num_seqs, min_segment_length, segmented_sequences, segmented_seq_lengths);
}

template<class T>
void normalize_queries_callback(T *query_values, size_t query_length, int normalization_mode, float* welford_mean, float* welford_ssq, long total_values_znormalized){
	load_and_normalize_queries(query_values, query_length, normalization_mode, welford_mean, welford_ssq, total_values_znormalized);
}

void free_results_callback(match_record** results){
	cudaFreeHost(*results);
}

int main(int argc, char** argv) {
	
	int verbose = 0;
	int help = 0;
	
	int signal_type = MEAN_SIGNAL;
	int rna = 0;
	int complement_only = 0;
	int complement = 0;
	
	int use_directory_as_subject = 0;

	int bed_file = 0;
	char* bed_filename;
	
	char *std_filename;
	
	int znorm = 0;
	int use_fast_anchor_calc = 0;
	bool use_std = false;
	int use_hard_dtw = 1;
	int minidtw_size = 10; // matching block size
	int minidtw_warp = 2;
	
	int match_max_ranks = 100; // Up to how many results do you want to show that meet the FDR and p value criteria?  Cannot be more than 2^20 (~1M)
	float match_max_pvalue = 0.01;
	float match_max_fdr = 1;
	float max_collinearity_dev = 0.25;
	int min_segment_length_sub = 0; // nanopore RNA segment length by default for the subject
	int min_segment_length_query = 0; // nanopore RNA segment length by default for the query
	
	std::string host = "localhost"; // Default minknow host
	int port = 8000; // Default minknow port
	int channel_begin = 1;
	int channel_end = 512;
	int pore_buff_size = 24000;
	int upper = 1300;
	int lower = 300;
	
	int instrand = 3;

	int num_devices;
	cudaGetDeviceCount(&num_devices);
	int num_threads = num_devices;
	int num_chan_threads = 1;
	
	// int fasta_file = 0;
	// char* fasta_filename;
	// int single_strand = 0;
	
	// int prefix = 0;
	// int minimal = 0;
	// int requested_suffix_length = 100;
	// int reverse = 0;

	// int input_is_binary = 0;
	int selection = 0; // 0 for positive, 1 for negative selection

	char c;

	char buff[FILENAME_MAX];
	GetCurrentDir( buff, FILENAME_MAX );
	std::string log_output_dir(buff);
	log_output_dir += "/log";

	while( ( c = getopt (argc, argv, "S:p:f:r:m:w:q:s:q:d:b:z:H:P:B:E:g:u:l:t:T:O:i:cCRnFDNvh") ) != -1 ) {
		switch(c) {		
			case 'S':
				if(optarg) use_directory_as_subject = atoi(optarg);
				break;
			case 'p':
				if(optarg) match_max_pvalue = atof(optarg);
				break;
			case 'f':
				if(optarg) match_max_fdr = atof(optarg);
				break;
			case 'r':
				if(optarg) match_max_ranks = atoi(optarg);
				break;
			case 'm':
				if(optarg) max_collinearity_dev = atof(optarg);
				break;
			case 'w':
				if(optarg) minidtw_warp = atoi(optarg);
				break;
			case 'a':
				if(optarg) minidtw_size = atoi(optarg);
				break;
			case 's':
				if(optarg) min_segment_length_sub = atoi(optarg);
				break;
			case 'q':
				if(optarg) min_segment_length_query = atoi(optarg);
				break;
			case 'd':
				if(optarg) std_filename = optarg;
				use_std = 1;
				break;
			case 'b':
				if(optarg) bed_filename = optarg;
				bed_file = 1;
				break;
			case 'z':
				if(optarg) znorm = atoi(optarg);
				break;
			case 'H':
				if(optarg) host = optarg;
				break;
			case 'P':
				if(optarg) port = atoi(optarg);
				break;
			case 'B':
				if(optarg) channel_begin = atoi(optarg); 
				break;
			case 'E':
				if(optarg) channel_end = atoi(optarg); 
				break;
			case 'g':
				if(optarg) pore_buff_size = atoi(optarg); 
				break;
			case 'u':
				if(optarg) upper = atoi(optarg); 
				break;
			case 'l':
				if(optarg) lower = atoi(optarg); 
				break;
			case 't':
				if(optarg) num_devices = atoi(optarg);
				if(num_devices > num_threads){
					std::cerr << "Number of threads specified (" << num_devices << ") exceeds number of GPUs on this system. Defaulting to " << num_threads << std::endl;
				}
				break;
			case 'T':
				if(optarg) num_chan_threads = atoi(optarg);
				break;
			// case 'A':
				// if(optarg) fasta_filename = optarg;
				// fasta_file = 1;
				// break;
			// case 'L':
				// requested_suffix_length = atoi(optarg);
				// break;
			case 'O':
				if(optarg) log_output_dir = optarg;
				break;
			case 'i':
				instrand = atoi(optarg);
				break;
			case 'c':
				complement = 1;
				break;
			case 'C':
				complement_only = 0;
				break;
			case 'R':
				rna = 1;
				break;
			case 'n':
				signal_type = STDDEV_SIGNAL;
				break;
			case 'F':
				use_fast_anchor_calc = 1;
				break;
			case 'D':
				use_hard_dtw = 0;
				break;
			// case 'M':
				// minimal = 1;
				// break;
			// case 'R':
				// reverse = 1;
				// break;
			// case 'y':
				// prefix = 1;
				// break;
			// case 'o':
				// single_strand = 1;
				// break;
			// case 'i':
				// input_is_binary = 1;
				// break;
			case 'N':
				selection = 1;
				break;
			case 'v':
				verbose = 1;
				break;
			case 'h':
				help = 1;
				break;	
			default:
				/* You won't actually get here. */
				break;
		}
	}
	
	int num_args = argc - optind;
	if (help || 
		num_args != 1){
		std::cerr << "Usage: " << argv[0] << " [options] <reference_genome>" << std::endl
					<< "Client that receives reads from a MinION device through MinKOWN and applies NVIDIA GPU (CUDA-enabled) accelerated Dynamic Time Warping on them." << std::endl
					<< "DTW implementation based on the FLASH DTW anchor co-linearity method." << std::endl
					<< "reference_genome is a subject file that's been indexed with magenta_short_index, which will be subject-scaled and cast to " << QTYPE_NAME << ")" << std::endl
					<< "Options for reading BED file:" << std::endl
					<< "[-b Read in BED file to compare matches to. Must provide file path with this option]" << std::endl << std::endl
					<< "Options relating to connecting and reading in values are:" << std::endl
					<< "[-H Host to open a connection on] default=" << host << std::endl
					<< "[-P Port to connect to] default=" << port << std::endl
					<< "[-B Start channel to get reads from] default=" << channel_begin << std::endl
					<< "[-E End channel to get reads from] default=" << channel_end << std::endl 
					<< "[-g Size of the buffers that will be used to store reads for each pore] default=" << pore_buff_size << std::endl
					<< "[-u Upper limit to check for which determines if a read is instrand] default=" << upper << std::endl
					<< "[-l Lower limit to check for which determines if a read is instrand] default=" << lower << std::endl
					<< "[-i The value in the subject file that states when data was in strand.] default= " << instrand << std::endl
					<< "[-t Number of threads reads will be processed on] default=" << num_threads << std::endl
					<< "[-T Number of threads to send and receive data to and from the MinKNOW] default=" << num_chan_threads << std::endl << std::endl
					<< "Options related to matches are:" << std::endl
					// << "[-i Input is binary]" << std::endl
					<< "[-p P-value limit for reporting matches (i.e. anchor DTW distance Mann-Whitney test random match probability for the subject DB used, floating point)] default=" << match_max_pvalue << std::endl
					<< "[-f FDR limit for reporting matches (i.e. Benjamini-Hochberg multiple-testing corrected p-value for the subject DB used, floating point)] default=" << match_max_fdr << std::endl
					<< "[-r ranks limit for reporting matches (i.e. the number of matches to report that pass the pvalue and FDR criteria. Guaranteed to be the best matches unless # passing criteria are > 2^20)] default=" << match_max_ranks << std::endl
					<< "[-m Warp max (proportion of length deviation allowed between query and subject in alignment, larger=more sensitive & longer runtime, floating point)> default=" << max_collinearity_dev << std::endl
					<< "[-s Minimum segment length subject (defines the minimum segment length that will be accepted for the subject. segmentation will not run if set to 0, int)] default=" << min_segment_length_sub << std::endl
					<< "[-q Minimum segment length query (defines the minimum segment length that will be accepted for the query. segmentation will not run if set to 0, int)] default=" << min_segment_length_query << std::endl << std::endl
					<< "[-N Negative selection (default is positive)]" << std::endl 
					<< "[-z Normalization type for z-norm of query against subject: 0 = NO_ZNORM, 1 = LOCAL_ZNORM, 2 = ONLINE_ZNORM, 3 = GLOBAL_ZNORM, 4 = PERCENTILE_ZNORM] default=" << znorm << std::endl
					<< "[-F Enable fast non-colinear-distances sampling algorithm (by default uses thorough sampling algorithm)]" << std::endl
					<< "[-d Enable standard deviation distance calculation (requires subject_std file as additional input)]" << std::endl
					<< "[-D Disable hard_dtw so that soft_dtw of size 10 and warp 2 may be used ('-M 10 -W 2' arguments)]" << std::endl
					<< "[-a Mini-DTW size, length of comparison between query and subject within a mini-DTW iteration] default=" << minidtw_size << std::endl
					<< "[-w Mini-DTW warp, boundary left and right of the diagonal in a mini-DTW matrix that the path may traverse] default=" << minidtw_warp << std::endl << std::endl
					// << "Options related to FastA indexing are" << std::endl
					// << "[-A index FastA file. Must provide file path with this option. NOTE: reference_genome argument would be the FastA file here]" << std::endl
					// << "[-o single strand indexing only]" << std::endl
					<< "[-n Use standard deviation for signal type in FastA files. Default uses mean.]" << std::endl
					<< "[-c also generate signal for the reverse complement strand]" << std::endl
					<< "[-C exclude default forward strand encoding]" << std::endl
					<< "[-R convert input as RNA (default is DNA)]" << std::endl << std::endl
					// << "Options related to FastA prefixing are" << std::endl
					// << "[-y prefix FastA sequences]" << std::endl
					// << "[-R reverse the sequence (e.g. for nanopore 3'->5' direct RNA analysis)]" << std::endl
					// << "[-L unique suffix length to include in the output (default " << requested_suffix_length << ")]" << std::endl
					// << "[-M minimal output]" << std::endl
					// << "Note: these options are only used when giving a FastA file as a reference. They will do nothing otherwise." << std::endl << std::endl
					<< "Options for logging are:" << std::endl
					<< "[-O Output directory for logs. Filename will be: [Year][Month][Day]_[Time]_ont_log.txt default directory is current working directory (" << log_output_dir << ")" << std::endl << std::endl
					<< "Additional options are:" << std::endl
					<< "[-v verbose mode]" << std::endl
					<< "[-h help (this message)]" << std::endl
					<< "Note: Reads will be obtained from the start channel to the end channel. End channel must not be smaller than start channel" << std::endl << std::endl;

		if(num_args > 1) std::cerr << "Error: Too many arguments." << std::endl;
		if(num_args < 1) std::cerr << "Error: No arguments given." << std::endl;

		return 0;
	}

	if(verbose) std::cerr << "Running in verbose mode:" << std::endl;
	ITree::interval_vector bed_intervals;
	if(bed_file && !populateITree(bed_intervals, bed_filename, verbose)){
		std::cerr << "Unable to populate ITree. Exiting." << std::endl;
		return 0;
	}

	time_t now = time(0);
    struct tm *tstruct = localtime(&now);

	int year = 1900 + tstruct->tm_year;
	int month = 1 + tstruct->tm_mon;
	int day = tstruct->tm_mday;
	int hour = 1 + tstruct->tm_hour;
	int minute = 1 + tstruct->tm_min;
	int second = 1 + tstruct->tm_sec;

	std::string log_output_name = std::to_string(year) + std::to_string(month) + std::to_string(day) + "_" + std::to_string(hour) + "-" + std::to_string(minute) + "-" + std::to_string(second) + "_ont_log.txt";

	std::string log_file_path = hasEnding(log_output_dir, FILE_SEPARATOR) ? log_output_dir + log_output_name : log_output_dir + FILE_SEPARATOR + log_output_name;
	if(verbose) std::cerr << "Output log path will be: " << log_file_path << std::endl;

	log_file.open(log_file_path, std::ios::out);
	if(verbose) std::cerr << "Running with " << num_chan_threads << " threads for sending/ receiving data and " << num_threads << " threads for running DTW." << std::endl;
	log_file << "Starting log: " << std::endl;
	signal(SIGINT, handler);

	// char *ref_filename;
	// if(!fasta_file){
		// ref_filename = argv[optind];
	// }

	// QTYPE *subject_values;
	// // Allow really big files
	// unsigned long long int num_subject_values = 0;
	// QTYPE *subject_stds;
	// unsigned long long int num_subject_stds;
	// if(!use_std)
		// subject_stds = 0;

	// std::vector< std::pair<size_t, char *> > subject_offsets;
	// subject_offsets.reserve(20);
	
	// Slurp the subject file up all at once. *subject_values will be dynamically allocated for us (free it ASAP please)
	// if(fasta_file){
		// if(!complement && complement_only){
			// std::cerr << "No strand was selected for encoding (i.e. both -c and -C, specified) aborting.\n";
			// return 2;
		// }
		// if(prefix){
			// std::vector<std::string> seq_prefixes;
			// if(!fasta2prefixes(fasta_filename, seq_prefixes, reverse, requested_suffix_length, minimal, verbose)){
				// std::cerr << "Unable to prefix sequences from " << fasta_filename << " so exiting." << std::endl;
				// return 0;
			// }
			// if(!populateSubjectWithFastAPrefix(&subject_values, &num_subject_values, seq_prefixes, subject_offsets, signal_type, rna, complement, complement_only, verbose)){
				// std::cerr << "Unable to populate subject with FastA prefixes. Aborting." << std::endl;
				// return 0;
			// }
		// }else{
			// if(!populateSubjectWithFastA(&subject_values, &num_subject_values, fasta_filename, subject_offsets, signal_type, rna, complement, complement_only, verbose)){
				// std::cerr << "Unable to populate subject with FastA file. Aborting." << std::endl;
				// exit(1);
			// }
		// }
	// }
	// else{
		// if(verbose) std::cerr << "Reading subject from file: " << ref_filename << std::endl;
		// std::string rawfile_name(std::string(ref_filename)+std::string(".hpr"));
		
		// if(read_data<QTYPE>(stringToChar(ref_filename), &subject_values, &num_subject_values)){
			// std::cerr << "Error while reading subject file, aborting" << std::endl;
			// return 3;
		// }
		// if(bed_file){
			// if(!load_subject_index(ref_filename, subject_offsets, &single_strand)){
				// std::cerr << "Could not load subject index for " << ref_filename << ", aborting." << std::endl;
				// exit(1);
			// }
		// }
		// // free(rawfile_name_c);
	// }
	
	char** ref_filenames;
	int num_files = 0;
	if(use_directory_as_subject){
		num_files = getAllFilesFromDir(argv[optind], &ref_filenames);
		if(num_files == 0){
			std::cerr << "Could not get any compatible files from " << argv[optind] << " so aborting." << std::endl;
			return 0;
		}
	} else{	// Only one file used as a subject
		ref_filenames = (char**)malloc(sizeof(char*));
		*ref_filenames = argv[optind];
		num_files = 1;
	}
	
	short strand_flags = 0;
	if(complement){
		strand_flags |= COMPLEMENT_STRAND;
	}
	if(!complement_only){
		strand_flags |= FORWARD_STRAND;
	}
	
	if(verbose) std::cerr << "Initializing GPU device" << std::endl;
	cudaSetDevice(0);									CUERR("Setting GPU device to be used");
	cudaDeviceReset();									CUERR("Resetting GPU device");
	
	QTYPE **subject_values = 0;
	size_t *num_subject_values = 0;
	QTYPE **subject_stds = 0;
	size_t *num_subject_stds = 0;
	char **sequence_names = 0;
	if(!use_std){
		cudaMallocHost(&subject_stds, sizeof(QTYPE*));	CUERR("Allocating memory for subject stds array");
		*subject_stds = 0;
	}
	std::vector< std::pair<size_t, char*> > subject_offsets;
	subject_offsets.reserve(20);
	
	// Slurp the subject file up all at once. *subject_values will be dynamically allocated for us (free it ASAP please)
	//------Get subject data------
	if(verbose) std::cerr << "Reading subject data from " << *ref_filenames << std::endl;
	int num_sequences_sub = read_data<QTYPE>(ref_filenames, num_files, &subject_values, &sequence_names, &num_subject_values, instrand, rna, signal_type, strand_flags);
	
	if(use_std) {
		if(verbose){
			std::cerr << "Reading subject_std from file " << std_filename << std::endl;
		}
		if(read_data<QTYPE>(&std_filename, 1, &subject_stds, &sequence_names, &num_subject_stds, 0, 0, 0, 0) == 0){
			std::cerr << "Error while reading subject std file, aborting" << std::endl;
			return 3;
		}
		if(*num_subject_values != *num_subject_stds) {
			std::cerr << "Error number of subject values does not match number of subject standard deviations, aborting" << std::endl;
			return 3;
		}
	} 

	size_t subject_length = 0;
	QTYPE* merged_subject = 0; 
	if(min_segment_length_sub > 0){
		QTYPE **segmented_sequences = 0;
		size_t *segmented_seq_lengths = 0;
		adaptive_segmentation<QTYPE>(subject_values, num_subject_values, num_sequences_sub, min_segment_length_sub, &segmented_sequences, &segmented_seq_lengths);
		merged_subject = merge_data(segmented_sequences, segmented_seq_lengths, subject_offsets, sequence_names, num_sequences_sub, &subject_length);
		
		if(verbose){
		  std::cerr << "Finished segmenting subject into " << subject_length << " values" << std::endl;
		  std::cerr << "Loading subject (" << subject_length << " values) to GPU" << std::endl;
		}
		std::cerr << std::endl;

		// cudaFreeHost(segmented_sequences);
		// cudaFreeHost(segmented_seq_lengths);
	} else{
		merged_subject = merge_data(subject_values, num_subject_values, subject_offsets, sequence_names, num_sequences_sub, &subject_length);
		if(verbose){
			std::cerr << "Loading subject (" << subject_length << " values) to GPU" << std::endl;
		}
		std::cerr << std::endl;
	}
	if(verbose) std::cerr << "Loading subject (" << num_subject_values << " values) to GPU" << std::endl;
	load_subject(merged_subject, *subject_stds, subject_length, use_std);
	
	// load_subject(subject_values, subject_stds, num_subject_values, use_std);
	// free(subject_values);
	// free(subject_stds);

	// Create a new connection
	// Connection new_con(host, port, verbose);

	// Create a client for the MinKNOW
	ReadUntilClient client(host, port, pore_buff_size, upper, lower, channel_begin, channel_end, verbose);
	
	flash_callback flash_callback_ptr= &flash_dtw_callback;
	aSegmentation_callback<QTYPE> adaptive_segmentation_ptr = &adaptive_segmentation_callback;
	LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr = &normalize_queries_callback;
	FreeResults_callback free_results_ptr = &free_results_callback;

	std::cerr << "Reads on " << num_threads << " threads:" << std::endl;
	client.ReadsRequest(flash_callback_ptr, adaptive_segmentation_ptr, normalize_queries_ptr, free_results_ptr, num_threads, num_chan_threads, 
									min_segment_length_query, max_collinearity_dev, match_max_pvalue, match_max_fdr, match_max_ranks, subject_offsets, 
									bed_intervals, selection, use_fast_anchor_calc, znorm, use_std, minidtw_size, minidtw_warp, use_hard_dtw, log_file, verbose);

	// Dealocate after running
	client.DeallocatePoreBuff(verbose);
	log_file.close();

	return 0;
}
