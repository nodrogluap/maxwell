#include "ReadUntilClient.h"
#include "Connection.h"

#include "minknow_api/acquisition.grpc.pb.h"
#include "minknow_api/analysis_configuration.grpc.pb.h"
#include "minknow_api/data.grpc.pb.h"
#include "minknow_api/device.grpc.pb.h"
#include "minknow_api/instance.grpc.pb.h"
#include "minknow_api/keystore.grpc.pb.h"
#include "minknow_api/log.grpc.pb.h"
#include "minknow_api/manager.grpc.pb.h"
#include "minknow_api/minion_device.grpc.pb.h"
#include "minknow_api/promethion_device.grpc.pb.h"
#include "minknow_api/protocol.grpc.pb.h"
#include "minknow_api/statistics.grpc.pb.h"

using namespace minknow_api::acquisition;
using namespace minknow_api::analysis_configuration;
using namespace minknow_api::data;
using namespace minknow_api::device;
using namespace minknow_api::instance;
using namespace minknow_api::keystore;
using namespace minknow_api::log;
using namespace minknow_api::manager;
using namespace minknow_api::minion_device;
using namespace minknow_api::promethion_device;
using namespace minknow_api::protocol;
using namespace minknow_api::statistics;

// Stubs for calling proto functions
std::unique_ptr<AcquisitionService::Stub> acq_stub_;
std::unique_ptr<AnalysisConfigurationService::Stub> ana_stub_;
std::unique_ptr<DataService::Stub> data_stub_;
std::unique_ptr<DeviceService::Stub> dev_stub_;
std::unique_ptr<InstanceService::Stub> ins_stub_;
std::unique_ptr<KeyStoreService::Stub> key_stub_;
std::unique_ptr<LogService::Stub> log_stub_;
std::unique_ptr<ManagerService::Stub> man_stub_;
std::unique_ptr<MinionDeviceService::Stub> min_stub_;
std::unique_ptr<PromethionDeviceService::Stub> prom_stub_;
std::unique_ptr<ProtocolService::Stub> prot_stub_;
std::unique_ptr<StatisticsService::Stub> stat_stub_;

struct ReadUntilClient::threadValues{
	float max_collinearity_dev, match_max_pvalue, match_max_fdr;
	int match_max_ranks, min_segment_length;
	std::vector< std::pair<size_t, char *> > subject_offsets;
	ITree::interval_vector bed_intervals;
	int selection, use_fast_anchor_calc, znorm;
	bool use_std;
	int minidtw_size, minidtw_warp;
	int use_hard_dtw;
};

template <typename T> class wqueue
{
	std::list<T> m_queue;
	pthread_mutex_t m_mutex;
	pthread_cond_t m_condv; 

	float pore_means[512] = { 0 };
	float pore_ssq[512] = { 0 };
	long pore_total_values[512] = { 0 };

	ReadUntilClient::threadValues thread_queue_values;

	ReadUntilClient* client;
	
	flash_callback flash_dtw_callback;
	aSegmentation_callback<QTYPE> adaptive_segmentation_callback;
	LoadNormalizeQueries_callback<QTYPE> normalize_queries_callback;
	FreeResults_callback free_results_callback;
	
	std::ofstream& log_file;
	int verbose;

	public:
		wqueue(ReadUntilClient::threadValues m_thread_queue_values, flash_callback m_flash_dtw_callback, aSegmentation_callback<QTYPE> m_adaptive_segmentation_callback, LoadNormalizeQueries_callback<QTYPE> m_normalize_queries_callback, FreeResults_callback m_free_results_callback, std::ofstream& m_log_file, ReadUntilClient* m_client, int m_verbose)
				: thread_queue_values(m_thread_queue_values), flash_dtw_callback(m_flash_dtw_callback), adaptive_segmentation_callback(m_adaptive_segmentation_callback), normalize_queries_callback(m_normalize_queries_callback), free_results_callback(m_free_results_callback), log_file(m_log_file), client(m_client), verbose(m_verbose) {
			// log_file = m_log_file;
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

		float* getPoreMeans() { return pore_means; };
		float* getPoreSSQ() { return pore_ssq; };
		long* getPoreTotalValues() { return pore_total_values; };

		// Additional functions to get the above variables
		int getSegLength() { return thread_queue_values.min_segment_length; }
		float getColDev() { return thread_queue_values.max_collinearity_dev; }
		float getMaxP() { return thread_queue_values.match_max_pvalue; }
		float getMaxFDR() { return thread_queue_values.match_max_fdr; }
		float getMaxRanks() { return thread_queue_values.match_max_ranks; }
		float getSelection() { return thread_queue_values.selection; }
		float getUseFastCalc() { return thread_queue_values.use_fast_anchor_calc; }
		float getZnorm() { return thread_queue_values.znorm; }
		float getUseSTD() { return thread_queue_values.use_std; }
		float getMiniDtwSize() { return thread_queue_values.minidtw_size; }
		float getMiniDtwWarp() { return thread_queue_values.minidtw_warp; }
		int useHardDtw() { return thread_queue_values.use_hard_dtw; }

		ReadUntilClient* getClient() { return client; }
		flash_callback getFlashDTWCallback() { return flash_dtw_callback; }
		aSegmentation_callback<QTYPE> getASegmentationCallback() { return adaptive_segmentation_callback; }
		LoadNormalizeQueries_callback<QTYPE> getLoadNormalizeCallback() { return normalize_queries_callback; }
		FreeResults_callback getFreeResultsCallback() { return free_results_callback; }
		
		std::ofstream& getLogFileStream() { return log_file; };

		std::vector< std::pair<size_t, char *> > getOffsets() { return thread_queue_values.subject_offsets; }
		ITree::interval_vector getIntervals() { return thread_queue_values.bed_intervals; }

		int getVerbose() { return verbose; }
};

class WorkItem
{
	
	QTYPE* m_reads;
	unsigned long long int m_num_reads;
	int m_read_num;
	uint32_t m_channel_num;
 
	public:
		WorkItem(QTYPE* reads, unsigned long long int num_reads, int read_num, uint32_t channel_num) 
			: m_reads(reads), m_num_reads(num_reads), m_read_num(read_num), m_channel_num(channel_num) {}
		~WorkItem() {}

		QTYPE* getReads() { return m_reads; }
		unsigned long long int getNumReads() { return m_num_reads; }
		int getReadNum() { return m_read_num; }
		uint32_t getChannelNum() { return m_channel_num; }
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
			int selection = m_queue.getSelection();
			bool use_std = m_queue.getUseSTD();
			int use_hard_dtw = m_queue.useHardDtw();

			float* pore_means = m_queue.getPoreMeans();
			float* pore_ssq = m_queue.getPoreSSQ();
			long* pore_total_values = m_queue.getPoreTotalValues();

			int znorm = m_queue.getZnorm();
			int min_segment_length = m_queue.getSegLength();
			ReadUntilClient* client = m_queue.getClient();

			// std::ofstream log_file = m_queue.getLogFileStream();
			for (int i = 0;; i++) {
				std::vector< std::pair<size_t, char *> > subject_offsets = m_queue.getOffsets();
				ITree::interval_vector bed_intervals = m_queue.getIntervals();
				WorkItem* item = m_queue.remove();
				
				QTYPE *query_values = item->getReads();
				// Allow really big files
				unsigned long long int num_query_values = item->getNumReads();
				
				LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr = m_queue.getLoadNormalizeCallback();

				if(min_segment_length > 0){
					if(verbose) std::cerr << "Segmenting reads" << std::endl;
					QTYPE **segmented_sequences = 0;
					size_t *segmented_seq_lengths = 0;					
					aSegmentation_callback<QTYPE> adaptive_segmentation_ptr = m_queue.getASegmentationCallback();
					(*adaptive_segmentation_ptr)(&query_values, (size_t *)&num_query_values, 1, min_segment_length, &segmented_sequences, &segmented_seq_lengths);
					// adaptive_segmentation<QTYPE>(&query_values, (size_t *)&num_query_values, 1, min_segment_length, &segmented_sequences, &segmented_seq_lengths);
					
					pore_total_values[item->getChannelNum()] += num_query_values;
			
					(*normalize_queries_ptr)(*segmented_sequences, (*segmented_seq_lengths), znorm, &pore_means[item->getChannelNum()], &pore_ssq[item->getChannelNum()], pore_total_values[item->getChannelNum()]);
					// load_and_normalize_queries<QTYPE>(*segmented_sequences, (*segmented_seq_lengths), znorm, &pore_means[item->getChannelNum()], &pore_ssq[item->getChannelNum()], pore_total_values[item->getChannelNum()]);
					if(verbose) std::cerr << "Finished segmenting query into " << *segmented_seq_lengths << " values" << std::endl;
					
					free(query_values);
					free(segmented_sequences);
					free(segmented_seq_lengths);
				} else{
					if(verbose) std::cerr << "Loading non-segmented query (" << num_query_values << " values) to GPU" << std::endl;

					(*normalize_queries_ptr)(query_values, num_query_values, znorm, &pore_means[item->getChannelNum()], &pore_ssq[item->getChannelNum()], pore_total_values[item->getChannelNum()]);
					// load_and_normalize_queries<QTYPE>(query_values, num_query_values, znorm, &pore_means[item->getChannelNum()], &pore_ssq[item->getChannelNum()], pore_total_values[item->getChannelNum()]);
	
				}
				query_values = 0;
				
				std::string query_name = std::to_string(item->getChannelNum()) + "_" + std::to_string(item->getReadNum()) + "_" + std::to_string(item->getNumReads());
				char* query_name_c = (char*)malloc(query_name.length());
				strcpy(query_name_c, query_name.c_str());

				match_record *results = 0;
				bool record_match_anchors = false;
				if(verbose) std::cerr << "Running FLASH DTW matching algorithm" << std::endl;
				// Passing 0 for first query arg means that it should use the ones that were loaded previously (e.g. segment_and_load_queries() call, or previous call to flash_dtw()
				flash_callback flash_callback_ptr = m_queue.getFlashDTWCallback();
				int num_results = (*flash_callback_ptr)(query_values, num_query_values, query_name_c, m_queue.getColDev(), m_queue.getMaxP(), m_queue.getMaxFDR(), m_queue.getMaxRanks(), znorm, &results, m_queue.getMiniDtwSize(), m_queue.getMiniDtwWarp(), record_match_anchors, m_queue.getUseFastCalc(), use_std, use_hard_dtw);
				// flash_dtw(query_values, num_query_values, query_name_c, m_queue.getColDev(), m_queue.getMaxP(), m_queue.getMaxFDR(), m_queue.getMaxRanks(), znorm, &results, &num_results, m_queue.getMiniDtwSize(), m_queue.getMiniDtwWarp(), record_match_anchors, m_queue.getUseFastCalc(), use_std, use_hard_dtw);
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
								// if(verbose) std::cerr << "Match #" << j << " met P value criteria but did not overlap any ranges in BED file, so rejecting." << std::endl;
								client->UnblockChannel(item->getChannelNum(), verbose);
								time_t now = time(0);
								struct tm *ltm = localtime(&now);
								m_queue.getLogFileStream() << 1 + ltm->tm_hour << ":"  << 1 + ltm->tm_min << ":" << 1 + ltm->tm_sec 
														   << " - Rejected Pvalue " << result.p_value << " at channel " << item->getChannelNum() << " due to no BED file overlap. Unblocking channel." << std::endl;
								pore_means[item->getChannelNum()] = 0;
								pore_ssq[item->getChannelNum()] = 0;
								pore_total_values[item->getChannelNum()] = 0;
							} else{	// Overlap found
								for (std::vector<Interval<size_t, std::string> >::const_iterator tree_it = tree_results.begin(); tree_it != tree_results.end(); ++tree_it){
									if(verbose) std::cerr << "Match #" << j << " met P value criteria and was found in BED file (" << (*tree_it).value << "), so accepting." << std::endl;
								}
							}
						} else { // No BED file read in so just accepting based on P value
							if(selection){	// negative selection
								// if(verbose) std::cerr << "P value (" << result.p_value << ") fell in desired rejection range, so rejecting match #" << j << "." << std::endl;
								client->UnblockChannel(item->getChannelNum(), verbose);
								time_t now = time(0);
								struct tm *ltm = localtime(&now);
								m_queue.getLogFileStream() << 1 + ltm->tm_hour << ":"  << 1 + ltm->tm_min << ":" << 1 + ltm->tm_sec
														   << " - Rejected Pvalue " << result.p_value << " at channel " << item->getChannelNum() << " becuase Pvalue fell in desired rejection range (negative selection). Unblocking channel." << std::endl;
								pore_means[item->getChannelNum()] = 0;
								pore_ssq[item->getChannelNum()] = 0;
								pore_total_values[item->getChannelNum()] = 0;
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
							time_t now = time(0);
							struct tm *ltm = localtime(&now);
							m_queue.getLogFileStream() << "Unblocking channel " << item->getChannelNum() 
													   << " at time " << 1 + ltm->tm_hour << ":"  << 1 + ltm->tm_min << ":" << 1 + ltm->tm_sec << std::endl;
							pore_means[item->getChannelNum()] = 0;
							pore_ssq[item->getChannelNum()] = 0;
							pore_total_values[item->getChannelNum()] = 0;
						}
					}
					m_queue.getLogFileStream() << "Match " << j << ": p-val " << results[j].p_value << ", fdr:" << results[j].fdr <<  ", matching blocks: " << results[j].num_anchors << ", rank: " << results[j].match_ordinal << std::endl << 
					results[j].left_anchor_query << "-" << results[j].right_anchor_query << "\t" << results[j].left_anchor_subject << "-" << results[j].right_anchor_subject << std::endl;
					m_queue.getLogFileStream() << std::flush;
				}
				if(query_values != 0)
					free(query_values);
				if(results){
					FreeResults_callback free_results_ptr = m_queue.getFreeResultsCallback();
					(*free_results_ptr)(&results);
				}
					//cudaFreeHost(results);
				
				delete item;
			}
		}
};

// Constructor that builds the ReadUntil Client
// Initializes the stubs of proto files needed for communicating with the MinKNOW
// channel is the connection to the MinKNOW
// buffer_size is the size of the buffers for each pore which will store events as they are read in
// upper is the upper bound matches will be searched against
// lower is the lower bound matches will be searches against
ReadUntilClient::ReadUntilClient(std::string host, int port, int buffer_size, int upper, int lower, int first_channel, int last_channel, int verbose){ 
																																  
	Connection new_con(host, port, verbose);
	std::shared_ptr<Channel> channel = new_con.get_channel();
	std::cerr << "Client connection established" << std::endl;
	
	acq_stub_ =  AcquisitionService::NewStub(channel);
	ana_stub_ = AnalysisConfigurationService::NewStub(channel);
	data_stub_ = DataService::NewStub(channel);
	dev_stub_ = DeviceService::NewStub(channel);
	ins_stub_ = InstanceService::NewStub(channel);
	key_stub_ = KeyStoreService::NewStub(channel);
	log_stub_ = LogService::NewStub(channel);
	man_stub_ = ManagerService::NewStub(channel);
	min_stub_ = MinionDeviceService::NewStub(channel);
	prom_stub_ = PromethionDeviceService::NewStub(channel);
	prot_stub_ = ProtocolService::NewStub(channel);
	stat_stub_ = StatisticsService::NewStub(channel);
	
	channel_begin = first_channel;
	std::cerr << "Max number of channels: ";
	int max_channels = GetMaxChannels(verbose);
	if(last_channel > max_channels){
		if(verbose) std::cerr << "Max number of channels (" << max_channels << ") is less than the one given (" << last_channel << ") so setting to " << max_channels << std::endl;
		channel_end = max_channels;
	} else{
		channel_end = last_channel;
	}

	std::cerr << "Version Info: "; 
	GetVersionInfo(verbose);

	std::cerr << "Status: ";
	GetStatus(verbose);

	std::cerr << "Output Directories: ";
	GetDirectories(verbose);
	
	if(verbose) std::cerr << "Setting upper and lower bounds for reads to be: [" << lower << ", " << upper << "]" << std::endl;
	upper_bound = upper;
	lower_bound = lower;
	if(verbose) std::cerr << "Initializing pore buffers with size: " << buffer_size << std::endl;
	for(int i = 0; i < 512; i++){
		pore_buffers[i] = (short*)malloc(sizeof(short) * buffer_size);
		// pore_ends[i] = 0;
		// pore_starts[i] = 0;
		// buffer_lengths[i] = 0;
	}
	pore_buff_size = buffer_size;
}

// Function that deallocates buffers initialized by the constructor
void ReadUntilClient::DeallocatePoreBuff(int verbose){	
	if(verbose) std::cerr << "Deallocating buffers:" << std::endl;
	for(int i = 0; i < 512; i++){
		free(pore_buffers[i]);
	}
}

// Function that receives a set of bytes and adds it to it's appropriate pore buffer
// reads is an array of bytes to be stored in a buffer
// channel_num is the channel number that the bytes were read from
// num_bytes is the size of the byte array
int ReadUntilClient::AddBitesToPoreBuffer(short* reads, int channel_num, int num_bytes, int verbose){

	// To state that we've received bytes that would extend past the size of the buffer, so we're overwritting bytes at the beginning of the array
	int overwritten = 0;	
	if(verbose) std::cerr << "Writing to pore buffer:" << std::endl;
	// Check of bytes to be written will extend past end of buffer
	if(num_bytes + pore_ends[channel_num-1] > pore_buff_size - 1){
		if(verbose) std::cerr << "Number of bytes to write to pore buffer wil exceed size of buffer. Overwritting beginning of buffer:" << std::endl; 
		// Number of bytes that extend past the buffer
		int num_leftover_bytes = num_bytes + pore_ends[channel_num-1] - (pore_buff_size - 1); 
		// Number of bytes that will be written to the end of the buffer
		int num_bytes_towrite = num_bytes - num_leftover_bytes; 

		// We're already at the end of the buffer
		if(num_leftover_bytes == num_bytes){ 
			std::copy(reads, reads + sizeof(short) * num_bytes, pore_buffers[channel_num]);
			// Set start of pore buffer to byte after overwritten bytes
			pore_starts[channel_num] = num_bytes;  
			pore_ends[channel_num] = num_bytes - 1;
		} else {
			// Copy bytes to the end of the buffer
			std::copy(reads, reads + sizeof(short) * num_bytes_towrite, pore_buffers[channel_num] + sizeof(short) * (pore_ends[channel_num] + 1)); 
			// Copy remaining bytes to beginning of the buffer
			std::copy(reads + sizeof(short) * (num_bytes_towrite + 1), reads + sizeof(short) * num_leftover_bytes, pore_buffers[channel_num]); 
			// Set start of pore buffer to byte after overwritten bytes
			pore_starts[channel_num] = num_leftover_bytes;  
			pore_ends[channel_num] = num_leftover_bytes - 1;
		}
		overwritten = 1;
	} else {
		if(pore_ends[channel_num] == 0)
			std::copy(reads, reads + sizeof(short) * num_bytes, pore_buffers[channel_num]);
		else
			std::copy(reads, reads + sizeof(short) * num_bytes, pore_buffers[channel_num] + sizeof(short) * (pore_ends[channel_num] + 1));
		pore_ends[channel_num] += num_bytes - 1;
	}
	return overwritten;
}

// Function that takes all pore buffers and puts the data into one larger buffer
// void AddAllBytesToBuffer(int verbose=0){
	// int start_of_copy = 0;
	// if(verbose) std::cerr << "Writing all bytes to buffer:" << std::endl;	
	// for(int i = 0; i < 512; i++){
		// if(pore_ends[i] < pore_starts[i]){
			// // Copy from start to end of the buffer
			// std::copy(pore_buffers[i] + sizeof(short) * pore_starts[i], pore_buffers[i] + sizeof(short) * (pore_buff_size - 1), byte_buffer + sizeof(short) * start_of_copy); 
			// start_of_copy += pore_buff_size - pore_starts[i] + 1;
			// // Copy from beginning of buffer to end
			// std::copy(pore_buffers[i] , pore_buffers[i] + sizeof(short) * pore_ends[i], byte_buffer + sizeof(short) * start_of_copy); 
			// start_of_copy += pore_ends[i] + 1;
			// buffer_lengths[i] = pore_buff_size;
		// } else {
			// std::copy(pore_buffers[i], pore_buffers[i] + sizeof(short) * pore_ends[i], byte_buffer + sizeof(short) * start_of_copy);
			// start_of_copy += pore_ends[i] + 1;
			// buffer_lengths[i] = pore_ends[i] + 1;
		// }
	// }
// }

// AcquisitionService

// Function that prints the status of the MinKNOW
void ReadUntilClient::GetStatus(int verbose){
	ClientContext context;
	::minknow_api::acquisition::CurrentStatusRequest request;
	::minknow_api::acquisition::CurrentStatusResponse response;
	::grpc::Status return_status = acq_stub_->current_status(&context, request, &response);
	::minknow_api::acquisition::MinknowStatus status = response.status();
	std::cerr << "Status is: " << status << std::endl;
}
	
// AnalysisConfigurationService

// DataService

// Function that creates a thread which sends requests to the MinKNOW
// first_channel: the first channel in the range to send requests for
// last_channel: the last channel in the range to send requests for
// num_con_threads: number of threads needed for the connections
// threadRequest_values: struct that contains all variables needed to run flash_dtw
// stream: the stream that requests will be sent over
void ReadUntilClient::ThreadRequest_Response(int first_channel, int last_channel, int num_con_threads, 
									ReadUntilClient::threadValues threadRequest_values, 
									flash_callback flash_callback_ptr, aSegmentation_callback<QTYPE> adaptive_segmentation_ptr, 
									LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr, FreeResults_callback free_results_ptr, std::ofstream& log_file, int verbose){


	if(verbose) std::cerr << "Starting thread" << std::endl;
	
	ClientContext context_data;
	
	// Create and send a data request
	::minknow_api::data::GetDataTypesRequest data_request;
	::minknow_api::data::GetDataTypesResponse data_response;
	::grpc::Status return_status = data_stub_->get_data_types(&context_data, data_request, &data_response);
	::minknow_api::data::GetDataTypesResponse_DataType_Type data_type;

	// Obtaining the data type based on how the data is setup 
	// TODO might not need this anymore
	if(data_response.has_uncalibrated_signal()){
		data_type = data_response.uncalibrated_signal().type();
	} else if(data_response.has_calibrated_signal()){
		data_type = data_response.calibrated_signal().type();
	}
	std::cerr << "Data type is: " << data_type << std::endl;
	
	ClientContext context_request;

	// Get interface for reading and writing to the reads stream
	std::shared_ptr< ::grpc::ClientReaderWriter<::minknow_api::data::GetLiveReadsRequest, ::minknow_api::data::GetLiveReadsResponse> > stream(data_stub_->get_live_reads(&context_request));

	// Build a request to get data from a channel that should be strand 
	::minknow_api::data::GetLiveReadsRequest request;
	::minknow_api::data::GetLiveReadsRequest::StreamSetup *client_setup = request.mutable_setup();
	client_setup->set_first_channel(first_channel);
	client_setup->set_last_channel(last_channel);
	client_setup->set_raw_data_type(GetLiveReadsRequest_RawDataType_UNCALIBRATED);
	client_setup->set_sample_minimum_chunk_size(0);
	// if(verbose) std::cerr << "Writing request for channels " << first_channel << " to " << last_channel << std::endl;
	stream->Write(request);
	// Output how long since last request was sent
	auto req_now = std::chrono::system_clock::now();
	auto req_now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(req_now);
	
	auto req_value = req_now_ms.time_since_epoch();
	long request_time = req_value.count();
	// long duration = request_time - time_req_start[state_data.channel() - 1];
	// if(time_req_start[state_data.channel() - 1] != 0){
		// float dur_sec = duration / 1000.f;
		// if(verbose) std::cerr << "Time since last request was sent for channel " << state_data.channel() << ": " << dur_sec << " seconds" << std::endl;
	// }
	// Don't send another request for this pore until it has been serviced and we have received a response for it
	for(int i = first_channel; i < last_channel; i++){ 
		pore_request_sent[i - 1] = true;
		time_req_start[i - 1] = request_time;
	}

	// Create the queue and consumer (worker) threads
	wqueue<WorkItem*> queue(threadRequest_values, flash_callback_ptr, adaptive_segmentation_ptr, normalize_queries_ptr, free_results_ptr, log_file, this, verbose);
	for(int i = 0; i < num_con_threads; i++){
		ConsumerThread* thread = new ConsumerThread(queue);
		thread->start();
	}

	// Create unique arenas for reads request and response
	google::protobuf::Arena arena;

	::minknow_api::data::GetLiveReadsResponse* response = google::protobuf::Arena::Create<::minknow_api::data::GetLiveReadsResponse>(&arena);

	// std::cerr << "Reading stream" << std::endl;
	// Get responses from MinKNOW
	while(stream->Read(response)){

		// std::cerr << "reading response" << std::endl;
		// if(verbose) std::cerr << "channel size: " << response->channels_size() << std::endl;
		::google::protobuf::Map< ::google::protobuf::uint32, ::minknow_api::data::GetLiveReadsResponse_ReadData > channel_reads = response->channels();

		// Loop to confirm we have the correct channel in our map
		// if(verbose){
			// std::cerr << "Channels in map are: ";
			// for(::google::protobuf::Map< ::google::protobuf::uint32, ::minknow_api::data::GetLiveReadsResponse_ReadData >::const_iterator it = channel_reads.begin(); it != channel_reads.end(); ++it){
				// std::cerr << it->first << ", ";
			// }
			// std::cerr << std::endl;
		// }

		for(::google::protobuf::Map< ::google::protobuf::uint32, ::minknow_api::data::GetLiveReadsResponse_ReadData >::const_iterator it = channel_reads.begin(); it != channel_reads.end(); ++it){
			// Get read data from the response
			::minknow_api::data::GetLiveReadsResponse_ReadData read = it->second;
			uint32_t channel = it->first;
			std::string id = read.id();
			uint64_t start = read.start_sample();	
			uint64_t chunk_start = read.chunk_start_sample();
			uint64_t chunk_len = read.chunk_length();
			uint32_t read_num = read.number();

			// Get time it took for the response to be returned after sending a request
			auto now = std::chrono::system_clock::now();
			auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
			
			auto res_value = now_ms.time_since_epoch();
			long response_time = res_value.count();
			// long duration = response_time - time_req_start[channel - 1];
			// float dur_sec = duration / 1000.f;
			// if(verbose) std::cerr << "Channel " << channel << " duration: " << dur_sec << " seconds" << std::endl;
			time_req_start[channel - 1] = response_time;

			// std::cerr << "Channel: " << channel 
					// << ", ID: " << id 
					// << ", Start: " << start 
					// << ", Chunk start: " << chunk_start 
					// << ", Chunk length: " << chunk_len 
					// << ", Read Number: " << read_num << std::endl;
			if(data_type == 0){
				unsigned long long int reads_length;
				// Convert bytes to a short array
				QTYPE* reads = BytetoArray <QTYPE> (read.raw_data(), &reads_length, verbose);
				// TODO: Send this to DTW algorithm
				// Check if the bytes we've read in are actually instrand based on bounds set by the user
				WorkItem* item;
				// if(ReadsInStrand <QTYPE> (reads, reads_length, verbose)){
					// std::cerr << "Printing results (short)" << std::endl;
					// for (int i = 0; i < reads_length; i++){
						// std::cerr << reads[i] << ", ";
					// }
					// std::cerr << std::endl;
					std::cerr << "Add to queue " << reads_length << std::endl;
					item = new WorkItem(reads, reads_length, read_num, channel);
					queue.add(item);
					sleep(1);
				// }
			}
			// Request has been serviced so we can start sending requests for this pore again
			pore_request_sent[channel - 1] = false;
		}
		
		
	}
	// Wait for the queue to be empty
	while (queue.size() > 0);
	std::cerr << "Done. Returning" << std::endl;
}

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
void ReadUntilClient::ReadsRequest(flash_callback flash_callback_ptr, aSegmentation_callback<QTYPE> adaptive_segmentation_ptr, LoadNormalizeQueries_callback<QTYPE> normalize_queries_ptr, 
														FreeResults_callback free_results_ptr, int num_con_threads, int num_chan_threads, int min_segment_length, float max_collinearity_dev, float match_max_pvalue, 
														float match_max_fdr, int match_max_ranks, std::vector< std::pair<size_t, char *> >& subject_offsets, ITree::interval_vector bed_intervals, int selection, 
														int use_fast_anchor_calc, int znorm, bool use_std, int minidtw_size, int minidtw_warp, int use_hard_dtw, std::ofstream& log_file, int verbose){

	// // Create unique arenas for reads request and response
	// google::protobuf::Arena arena;

	std::cerr << "Getting reads for channels in the range of " << channel_begin << " to " << channel_end << ":" << std::endl;

	// ::minknow_api::data::GetLiveReadsResponse* response = google::protobuf::Arena::Create<::minknow_api::data::GetLiveReadsResponse>(&arena);

	std::cerr << "Creating threads" << std::endl;
	int num_channels = channel_end - channel_begin + 1; // +1 to include the first channel (channels 1 - 16 should include 16 threads and not 15, for example)
	int num_channels_per_thread = num_channels / num_chan_threads;
	std::cerr << "Num channels per thread: " << num_channels_per_thread << std::endl;
	// std:thread channel_threads[num_threads];
	std::vector<std::thread> channel_threads;
	channel_threads.reserve(num_chan_threads);
	int start_channel = channel_begin;
	int end_channel = 0;

	ReadUntilClient::threadValues threadRequest_values;
	threadRequest_values.min_segment_length = min_segment_length;
	threadRequest_values.max_collinearity_dev = max_collinearity_dev;
	threadRequest_values.match_max_pvalue = match_max_pvalue;
	threadRequest_values.match_max_fdr = match_max_fdr;
	threadRequest_values.match_max_ranks = match_max_ranks;
	threadRequest_values.subject_offsets = subject_offsets;
	threadRequest_values.bed_intervals = bed_intervals;
	threadRequest_values.selection = selection;
	threadRequest_values.use_fast_anchor_calc = use_fast_anchor_calc;
	threadRequest_values.znorm = znorm;
	threadRequest_values.use_std = use_std;
	threadRequest_values.minidtw_size = minidtw_size;
	threadRequest_values.minidtw_warp = minidtw_warp;
	threadRequest_values.use_hard_dtw = use_hard_dtw;

	for(int i = 0; i < num_chan_threads; i++){
		if(i == num_chan_threads - 1 && start_channel + num_channels_per_thread - 1 < channel_end){	// Last thread for any remaining channels
			end_channel = channel_end;					
		} else{
			end_channel = start_channel + num_channels_per_thread - 1;
		}
		if(verbose) std::cerr << "Creating thread for channels " << start_channel << " to " << end_channel << std::endl;

		channel_threads.emplace_back(&ReadUntilClient::ThreadRequest_Response, this, start_channel, end_channel, num_con_threads, threadRequest_values, flash_callback_ptr, adaptive_segmentation_ptr, normalize_queries_ptr, free_results_ptr, std::ref(log_file), verbose);
		start_channel += num_channels_per_thread;
	}

	std::cerr << "All threads launched" << std::endl;

	for (std::vector<std::thread>::iterator it = channel_threads.begin(); it != channel_threads.end(); ++it){
		(*it).join();
	}
	// std::thread threadObj(&ReadUntilClient::ThreadRequests, this, first_channel, last_channel, stream, verbose);

	
	// threadObj.join();
	// grpc::Status status = stream->Finish();
	// if(!status.ok()){
		// std::cerr << "RPC failed." << std::endl;
	// }
}
	
// DeviceService

void ReadUntilClient::UnblockChannel(uint32_t channel_num, int verbose){
	ClientContext context;
	::minknow_api::device::UnblockRequest request;
	::minknow_api::device::UnblockResponse response;
	request.add_channels(channel_num);
	::grpc::Status return_status = dev_stub_->unblock(&context, request, &response);
	if(verbose) std::cerr << "Unblocking channel " << channel_num << std::endl;
}

int ReadUntilClient::GetMaxChannels(int verbose){
	ClientContext context;
	::minknow_api::device::GetFlowCellInfoRequest request;
	::minknow_api::device::GetFlowCellInfoResponse response;
	::grpc::Status return_status = dev_stub_->get_flow_cell_info(&context, request, &response);
	
	::google::protobuf::uint32 max_channel = response.channel_count();
	
	if(verbose) std::cerr << max_channel << std::endl;
	return response.channel_count();
}

// InstanceService

// Function that prints the directories output for the run will be saved to
void ReadUntilClient::GetDirectories(int verbose){
	ClientContext context;
	::minknow_api::instance::GetOutputDirectoriesRequest request;
	::minknow_api::instance::OutputDirectories response;
	::grpc::Status return_status = ins_stub_->get_output_directories(&context, request, &response);
	std::cerr << response.output() << std::endl;
}

// Function that prints the version info for the MinKNOW
void ReadUntilClient::GetVersionInfo(int verbose){
	ClientContext context;
	::minknow_api::instance::GetVersionInfoRequest request;
	::minknow_api::instance::GetVersionInfoResponse response;
	::grpc::Status return_status = ins_stub_->get_version_info(&context, request, &response);
	std::cerr << response.minknow().full() << std::endl;
}
	
// KeyStoreService

// LogService

// ManagerService

// MinionDeviceService

// PromethionDeviceService

// ProtocolService

// StatisticsService

