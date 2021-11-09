/*
	wqueue.h

	Worker thread queue based on the Standard C++ library list
	template class.

	This file has been modified to fit the requirements of the ont_simple_client system.

	------------------------------------------

	Copyright (c) 2013 Vic Hargrave

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

*/

#ifndef __wqueue_h__
#define __wqueue_h__

#if defined(_WIN32)
	#include "pthread.h"
#else
	#include <pthread.h>
#endif

#include "cuda_utils.h" // CUERR()


#define QTYPE short
#define QTYPE_ACC int
/*
// The search algorithm 
#include "flash_dtw.cuh"
// Functions to read in data files (non-GPU based) 
#include "flash_utils.hpp"
*/
// For UI message purposes
#define QTYPE_NAME "short"
#include "thread.h"
#include "IntervalTree.h"
#include "ReadUntilClient.cuh"

typedef IntervalTree<size_t, std::string> ITree;

#include <list>

using namespace std;

template <typename T> class wqueue
{
	list<T> m_queue;
	pthread_mutex_t m_mutex;
	pthread_cond_t m_condv; 

	// Additional variables needed for flash DTW
	float avg_segment_size;
	float segment_split_criterion;
	float max_collinearity_dev;
	float match_max_pvalue;
	float match_max_fdr;
	int match_max_ranks;
	std::vector< std::pair<size_t, char *> > subject_offsets;
	ITree::interval_vector bed_intervals;
	int selection;
	ReadUntilClient::ReadUntilClient* client;
	int verbose;

	public:
		wqueue(float avg_seg_size, float seg_split_cri, float max_col_dev, float match_max_p, float mat_max_fdr, 
			   int mat_max_ranks, std::vector< std::pair<size_t, char *> >& sub_offsets, ITree::interval_vector bed_inter, int m_selection, 
			   ReadUntilClient* m_client, int m_verbose)
				: avg_segment_size(avg_seg_size), segment_split_criterion(seg_split_cri), max_collinearity_dev(max_col_dev),
				  match_max_pvalue(match_max_p), match_max_fdr(mat_max_fdr), match_max_ranks(mat_max_ranks),
				  subject_offsets(sub_offsets), bed_intervals(bed_inter), selection(m_selection), client(m_client), verbose(m_verbose) {
			pthread_mutex_init(&m_mutex, NULL);
			pthread_cond_init(&m_condv, NULL);
		}
		~wqueue() {
			pthread_mutex_destroy(&m_mutex);
			pthread_cond_destroy(&m_condv);
		}
		void add(T item) {
			pthread_mutex_lock(&m_mutex);
			m_queue.push_back(item);
			pthread_cond_signal(&m_condv);
			pthread_mutex_unlock(&m_mutex);
		}
		T remove() {
			pthread_mutex_lock(&m_mutex);
			while (m_queue.size() == 0) {
				pthread_cond_wait(&m_condv, &m_mutex);
			}
			T item = m_queue.front();
			m_queue.pop_front();
			pthread_mutex_unlock(&m_mutex);
			return item;
		}
		int size() {
			pthread_mutex_lock(&m_mutex);
			int size = m_queue.size();
			pthread_mutex_unlock(&m_mutex);
			return size;
		}

		// Additional functions to get the above variables
		float getSegSize() { return avg_segment_size; }
		float getSplitCri() { return segment_split_criterion; }
		float getColDev() { return max_collinearity_dev; }
		float getMaxP() { return match_max_pvalue; }
		float getMaxFDR() { return match_max_fdr; }
		float getMaxRanks() { return match_max_ranks; }

		std::vector< std::pair<size_t, char *> > getOffsets() { return subject_offsets; }
		ITree::interval_vector getIntervals() { return bed_intervals; }

		int getVerbose() { return verbose; }
};

class WorkItem
{
	
	QTYPE* m_reads;
	unsigned long long int m_num_reads;
	int m_read_num;
	uint32_t m_channel_num;
 
	public:
		WorkItem(short* reads, unsigned long long int num_reads, int read_num, uint32_t channel_num) 
			: m_reads(reads), m_num_reads(num_reads), m_read_num(read_num), m_channel_num(channel_num) {}
		~WorkItem() {}

		QTYPE* getReads() { return m_reads; }
		unsigned long long int getNumReads() { return m_num_reads; }
		int getReadNum() { return m_read_num; }
		uint32_t getChannelNum() { return channel_num; }
};

class ConsumerThread : public Thread
{
	wqueue<WorkItem*>& m_queue;

	public:
		ConsumerThread(wqueue<WorkItem*>& queue) : m_queue(queue) {}

		void* run() {
			// Remove 1 item at a time and process it. Blocks if no items are 
			// available to process.
			int verbose = m_queue.getVerbose();
			for (int i = 0;; i++) {
				std::vector< std::pair<size_t, char *> > subject_offsets = m_queue.getOffsets();
				ITree::interval_vector bed_intervals = m_queue.getIntervals();
				WorkItem* item = m_queue.remove();
				
				QTYPE *query_values = item->getReads();
				// Allow really big files
				unsigned long long int num_query_values = item->getNumReads();

				if(verbose) std::cerr << "Segmenting reads" << std::endl;
				segment_and_load_queries(query_values, (unsigned int *) &num_query_values, 1, m_queue.getSegSize(), m_queue.getSplitCri(), GLOBAL_ZNORM);
				free(query_values);
				
				match_record *results;
				int num_results;
				bool record_match_anchors = false;
				if(verbose) std::cerr << "Running FLASH DTW matching algorithm" << std::endl;
				// Passing 0 for first query arg means that it should use the ones that were loaded previously (e.g. segment_and_load_queries() call, or previous call to flash_dtw()
				/*
				flash_dtw(query_values, num_query_values, m_queue.getColDev(), m_queue.getMaxP(), m_queue.getMaxFDR(), m_queue.getMaxRanks(), GLOBAL_ZNORM, &results, &num_results, record_match_anchors);
				// The memory dynamically allocated by flash_dtw() is pinned physical memory allocated with cudaMallocHost(), so be sure to use cudaFreeHost() rather than free() to avoid a memory leak
				if(verbose) std::cerr << "Number of matches: " << num_results << std::endl;
				for(int j = 0; j < num_results; j++){
					match_record result = results[j];
					if(result.p_value <= m_queue.getMaxP()){
						if(!bed_intervals.empty()){ // A BED file was read in
							// Get location of result in the idx file and check for overlap in the vector
							size_t start = 0, end = 0;
							size_t result_start = 0, result_end = 0;
							for(std::vector< std::pair<size_t, char *> >::iterator offset_it = subject_offsets.begin(); offset_it != subject_offsets.end(); ++offset_it) {
								if(end == 0){
									start = 0;
								} else{
									start = end + 1;
								}
								end = (*offset_it).first;
								if(result.left_anchor_subject >= start && result.right_anchor_subject <= end){
									result_start = result.left_anchor_subject - start; // Get the actual start and end values of the results
									result_end = result.right_anchor_subject - start;
									break;
								}
							}
							ITree bed_tree = ITree(std::move(bed_intervals));
							auto tree_results = bed_tree.findOverlapping(result_start, result_end);
							if(tree_results.size() == 0){ // No overlaps found
								if(verbose) std::cerr << "Match #" << j << " met P value criteria but did not overlap any ranges in BED file, so rejecting." << std::endl;
								// Decide whether to continue reading from pore or not
							} else{
								for (std::vector<Interval<size_t, std::string> >::const_iterator tree_it = tree_results.begin(); tree_it != tree_results.end(); ++tree_it){
									if(verbose) std::cerr << "Match #" << j << " met P value criteria and was found in BED file (" << (*tree_it).value << "), so accepting." << std::endl;
									// Decide whether to continue reading from pore or not
								}
							}
						} else { // No BED file read in so just accepting based on P value
							if(selection){	// negative selection
								if(verbose) std::cerr << "P value (" << result.p_value << ") fell in desired rejection range, so rejecting match #" << j << "." << std::endl;
								client->UnblockChannel(item->getChannelNum(), verbose);
							} else { // positive selection
								if(verbose) std::cerr << "Match #" << j << " accepted because P value (" << result.p_value << ") was what we wanted." << std::endl;
							}
						}
					} else{ // P value wasn't what we wanted
						if(selection){	// negative selection
							if(verbose) std::cerr << "P value (" << result.p_value << ") did not fall in desired rejection range, so accepting match #" << j << "." << std::endl;
						} else { // positive selection
							if(verbose) std::cerr << "Match #" << j << " rejected because P value (" << result.p_value << ") wasn't what we wanted." << std::endl;
							client->UnblockChannel(item->getChannelNum(), verbose);
						}
					}
					std::cerr << "Match " << j << ": p-val " << results[j].p_value << ", fdr:" << results[j].fdr <<  ", matching blocks: " << results[j].num_anchors << ", rank: " << results[j].match_ordinal << std::endl << 
					results[j].left_anchor_query << "-" << results[j].right_anchor_query << "\t" << results[j].left_anchor_subject << "-" << results[j].right_anchor_subject << std::endl;
				}
				*/
				if(results){
					cudaFreeHost(results);
				}
				delete item;
			}
		}
};

#endif
