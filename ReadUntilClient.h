#ifndef _READUNTIL_H
#define _READUNTIL_H

#include <string>
#include <sstream>
#include <memory>

#if defined(_WIN32)
	#include <windows.h>
	#define sleep(x) Sleep(1000 * (x))
	#include "pthread.h"
#else
	#include <unistd.h>
	#include <pthread.h>
#endif

#include <stdio.h>

#include <thread>
#include <iostream>
#include <fstream>
#include <map>
#include <iterator>
#include <vector>
#include <exception>
#include <cstdlib>
#include <stdlib.h>
#include <chrono>
#include <time.h>

//TODO: This might need to be moved or set up differently
#define QTYPE float
#define QTYPE_ACC_float
//#define QTYPE_short
//#define QTYPE_ACC_short
//#define QTYPE_half
//#define QTYPE_ACC_half

#include "IntervalTree.h"

#include "include/cdtw.h"
#include "algo_datatypes.h"

#define MINIDTW_STRIDE 1 // can only be one of {1, 2, 4, 8}

// Functions to read in data files (non-GPU based)
#include "all_utils.hpp"
// For UI message purposes
#define QTYPE_NAME "single-precision floating point"

#include "thread.h"

#include <list>

typedef IntervalTree<size_t, std::string> ITree;
using flash_callback = int (*)(QTYPE*, unsigned long long int, char*, float, float, float, float, float, match_record**, float, float, bool, float, bool, int);
template<typename T>
using aSegmentation_callback = void (*)(T **, size_t *, int , int , T ***, size_t **);
template<class T>
using LoadNormalizeQueries_callback = void (*) (T* , size_t, int, float*, float*, long);
using FreeResults_callback = void (*)(match_record**);

class ReadUntilClient {

	public:

		ReadUntilClient();

		struct threadValues;

		// Constructor that builds the ReadUntil Client
		// Initializes the stubs of proto files needed for communicating with the MinKNOW
		// channel is the connection to the MinKNOW
		// buffer_size is the size of the buffers for each pore which will store events as they are read in
		// upper is the upper bound matches will be searched against
		// lower is the lower bound matches will be searches against
		ReadUntilClient(std::string host, int port, int buffer_size, int upper, int lower, int start_channel, int end_channel, int verbose=0);

		// Function that deallocates buffers initialized by the constructor
		void DeallocatePoreBuff(int verbose=0);

		// Function that receives a set of bytes and adds it to it's appropriate pore buffer
		// reads is an array of bytes to be stored in a buffer
		// channel_num is the channel number that the bytes were read from
		// num_bytes is the size of the byte array
		int AddBitesToPoreBuffer(short* reads, int channel_num, int num_bytes, int verbose=0);

		// Function that takes all pore buffers and puts the data into one larger buffer
		// void AddAllBytesToBuffer(int verbose=0);

		// AcquisitionService

		// Function that prints the status of the MinKNOW
		void GetStatus(int verbose=0);
	
		// AnalysisConfigurationService
	
		// DataService

		// Function to unblock the channel requested
		// channel_num is the channel requested
		void UnblockChannel(uint32_t channel_num, int verbose=0);
		
		// Function to get the max number of channels in the MinION
		int GetMaxChannels(int verbose=0);

		// Function that creates a thread which sends requests to the MinKNOW
		// first_channel: the first channel in the range to send requests for
		// last_channel: the last channel in the range to send requests for
		// num_con_threads: number of threads needed for the connections
		// threadRequest_values: struct that contains all variables needed to run flash_dtw
		// data_type: the type of the data that will be read in
		// stream: the stream that requests will be sent over
		void ThreadRequest_Response(int first_channel, int last_channel, int num_con_threads, 
									ReadUntilClient::threadValues threadRequest_values, 
									flash_callback flash_callback_ptr, aSegmentation_callback<QTYPE> adaptive_segmentation_ptr, 
									LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr, FreeResults_callback free_results_ptr, std::ofstream& log_file, int verbose = 0);

		// Function that gets reads from the MinKNOW from pores defined by the range [first_channel, last_channel]
		// first_channel is the first channel we want to search for states on
		// last_channel is the last channel we want to search for states on
		// num_con_threads is the number of threads we will be creating to run the queue on
		// avg_segment_size is the segment size mean
		// segment_split_criterion is the attenuation limit for a segment
		// max_collinearity_dev is the warp max
		// match_max_pvalue is the P-value limit for reporting matches
		// match_max_fdr is the FDR limit for reporting matches
		// match_max_ranks is the ranks limit for reporting matches
		// subject_offsets are the ranges of the sequences that were read in from the reference file
		// bed_intervals are the intervals read in from the bead file
		// selection is the type of selection that will be used (positive/ negative)
		// use_fast_anchor_calc is a flag used for determining which function will be used for anchor calculations
		// znorm determines which type of znormalization will be used
		// use_std is a flag that determines if we are using the standard deviation of the subject
		// use_adaptive is a flag that checks if we're using adaptive segmentation
		// minidtw_size is the size of the mini dtw length
		// minidtw_warp is the warp for mini dtw
		// log_file is the file that logs will be written to
		void ReadsRequest(flash_callback flash_callback_ptr, aSegmentation_callback<QTYPE> adaptive_segmentation_ptr, LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr, 
										FreeResults_callback free_results_ptr, int num_con_threads, int num_chan_threads, int min_segment_length_query, float max_collinearity_dev, float match_max_pvalue, 
										float match_max_fdr, int match_max_ranks, std::vector< std::pair<size_t, char *> >& subject_offsets, ITree::interval_vector bed_intervals, int selection, int use_fast_anchor_calc, 
										int znorm, bool use_std, int minidtw_size, int minidtw_warp, int use_hard_dtw, std::ofstream& log_file, int verbose=0);
	
		// DeviceService
	
		// InstanceService
	
		// Function that prints the directories output for the run will be saved to
		void GetDirectories(int verbose=0);

		// Function that prints the version info for the MinKNOW
		void GetVersionInfo(int verbose=0);
	
		// KeyStoreService
	
		// LogService
	
		// ManagerService
	
		// MinionDeviceService

		// PromethionDeviceService
	
		// ProtocolService
	
		// StatisticsService
	
	private:

		// Buffers that store individual pore data
		short* pore_buffers[512];
		int pore_buff_size;
		int pore_ends[512] ={0};
		int pore_starts[512] ={0};

		// Bounds for pore range
		int lower_bound;
		int upper_bound;
		
		// Channel range
		int channel_begin;
		int channel_end;

		// Keep track of if a request has been sent and how long it's taking to be serviced
		bool pore_request_sent[512] = {false};
		long time_req_start[512] = {0};
		
		// Function that converts an array of bytes to a T array
		// bytes: the original byte array
		// data_size: the size of the byte array
		// returns the converted array
		template <class T>
		T* BytetoArray(std::string bytes, unsigned long long int* data_size, int verbose=0){
			unsigned char* buffer = new unsigned char[bytes.length()];
			memcpy(buffer, bytes.data(), bytes.length());
			(*data_size) = (bytes.length() * sizeof(buffer[0])) / sizeof(T);
			T* data = (T*)buffer;
			return data;
		}

		// Function that determines if the data being read in is in strand based on predetermined range
		// array: the array of data to be looked at
		// size_of_array: the size of the array		
		// returns true if in strand, false if not
		template <class T>
		bool ReadsInStrand(T *array, int size_of_array, int verbose=0){
			float average = 0;
			int count = 0;
			bool in_strand = false;
			// Itterate through array
			for(int i = 0; i < size_of_array; i++){	
				// Get 10th of array for average
				float tenth = array[i] / 10;	
				// Get 10th of average to remove
				float tenth_of_avg = average / 10;	
				// Add 10th of array to average
				average += tenth;
				// Check for if we've averaged 10 values yet
				if(count >= 10){	
					// Remove 10th of average before addition
					average -= tenth_of_avg;
					// Check if average falls within the range we're looking for
					if(average >= lower_bound && average <= upper_bound){	
						in_strand = true;	
						break;	
					}
				} else
					count++;	// Add to counter if we haven't averaged 10 values yet
			}
			return in_strand;
		}
};

#endif
