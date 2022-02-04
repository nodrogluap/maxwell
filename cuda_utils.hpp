#ifndef __dba_cuda_utils_included
#define __dba_cuda_utils_included

#include "ont_utils.cuh"
#include "all_utils.hpp"

#define CUDA_THREADBLOCK_MAX_L1CACHE 48000
// Note that you should not change this to >1028 unless you carefully review all the code for reduction steps that imply 32x32 map-reduce!
#if defined(_DEBUG_REG)
#define CUDA_THREADBLOCK_MAX_THREADS 256
#else
#define CUDA_THREADBLOCK_MAX_THREADS 1024
#endif
#define CUDA_WARP_WIDTH 32
#define CUERR(MSG) { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << MSG << ")" << std::endl; exit((int) err);}}
#define FULL_MASK 0xffffffff

#define DIV_ROUNDUP(numerator, denominator) (((numerator) + (denominator) - 1)/(denominator))

// Finds the minimum value in a warp (here of 32) and returns the result
// val - the values in the warp that will be compared with one another
// returns the value found to be the minimum in the warp
template<typename T>
__inline__ __device__ T warpReduceMin(T val){
    for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
        T tmpVal = __shfl_down_sync(FULL_MASK, val, offset);
        if (tmpVal < val){
            val = tmpVal;
        }
    }
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val){
    for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
        T tmpVal = __shfl_down_sync(FULL_MASK, val, offset);
        if (tmpVal > val){
            val = tmpVal;
        }
    }
    return val;
}

// Function that converts a short to the given template value
// data - the short buffer to be converted
// data_length - the length of the buffer passed in
// returns a new buffer that is of type template with the short data stored in it
template <class T>
T* cuda_shortToTemplate(short* data, long long data_length){
	T* return_data;
	cudaMallocHost(&return_data, sizeof(T)*data_length);	CUERR("Cannot allocate CPU memory for FAST5 signal");
	std::transform(data, data + data_length, return_data, [](short s){ return (T)s; });
	return return_data;
}

//-------------------------
//-------------------------
//-------READ DATA---------
//-------------------------
//-------------------------

// Function that reads in data from a text file and stores it into a buffer
// text_file_name - the file we are reading from
// output_vals - the buffer we are storing the data to
// num_output_vals - the number of values stored in the buffer
// returns 1 on a success, 0 otherwise
template <typename T>
int
read_text_data(const char *text_file_name, T **output_vals, size_t *num_output_vals){

	// Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for output_vals
	// std::ios::sync_with_stdio(false); //optimization
	const int SZ = 1024 * 1024;
	std::vector <char> read_buffer( SZ );
	std::ifstream ifs(text_file_name); 
	if(!ifs){
		std::cerr << "Error reading in file " << text_file_name << ", skipping" << std::endl;
		return 0;
	}
	int n = 0;
	while(int sz = FileRead(ifs, read_buffer)) {
		n += CountLines(read_buffer, sz);
	}
	*num_output_vals = n;
	if(n == 0){
		std::cerr << "File " << text_file_name << " is empty or not properly formatted. Skipping." << std::endl;
		ifs.close();
		return 0;
	}

	T *out = 0;
	cudaMallocHost(&out, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from text file");
	
	// Read the actual values
	ifs.clear(); // get rid of EOF error state
	ifs.seekg(0, std::ios::beg);
	std::stringstream in; // Make a stream for the line itself
	std::string line;
	int i = 0;
	while(n--){	// Read line by line
		std::getline(ifs, line); in.str(line);
		in >> out[i++]; // Read the first whitespace-separated token
		in.clear(); // to reuse the stringatream parser
	}

	// Only set the output if all the data was succesfully read in.
	*output_vals = out;
	ifs.close();
	return 1;
}

#if HDF5_SUPPORTED == 1

// Function that reads sequence data from a fast5 file and stores it into a buffer as well as the names of each sequence read in
// fast5_file_name - the file we are reading from
// sequences - the buffer that will store the fast5 data
// sequence_names - the buffer that will store the names of each sequence in the fast5 file
// sequence_lengths - the length of each sequence stored in the buffer
// returns the number of sequences read in
template <typename T>
int
read_fast5_data(const char *fast5_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	
	int local_seq_count_so_far = 0;

	hid_t file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){ // No message, assume scan function called earlier provided these
		return 0;
	}
	bool old_format = true;
	H5Eset_auto1(NULL, NULL);
	// Old format, one read per file
	hid_t read_group = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
	if(read_group < 0){ // New formst, multiple reads per file
		read_group = H5Gopen(file_id, "/", H5P_DEFAULT);
		old_format = false;
	}
	hsize_t num_read_objects = 0;
	if(read_group < 0 || H5Gget_num_objs(read_group, &num_read_objects)){
		H5Gclose(read_group);
		H5Fclose(file_id);
		return 0;
	}
	char *read_subgroup_name = NULL;
	for(int i = 0; i < num_read_objects; ++i){
		ssize_t name_size = H5Gget_objname_by_idx(read_group, i, NULL, 0);
		if(name_size < 5){
			continue; // Too short to be "[Rr]ead_..."
		}
		name_size++; // add space for NULL termination
		errno = 0;
		read_subgroup_name = (char *) (read_subgroup_name == NULL ? std::malloc(name_size) : std::realloc(read_subgroup_name, name_size));
	
		if(errno || (read_subgroup_name == NULL)){
			std::cerr << "Error in malloc/realloc for HDF5 read group name: " << strerror(errno) << std::endl;
			exit(FAST5_CANNOT_MALLOC_READNAME);
		}
		name_size = H5Gget_objname_by_idx(read_group, i, read_subgroup_name, name_size);
		// Should have the form Read_# (old) or read_????? (new)
		if(name_size < 5 || (read_subgroup_name[0] != 'R' && read_subgroup_name[0] != 'r') || read_subgroup_name[1] != 'e' || read_subgroup_name[2] != 'a' || read_subgroup_name[3] != 'd' || read_subgroup_name[4] != '_'){
			std::cerr << "Skipping " << read_subgroup_name << " as it does not follow the naming convention" << std::endl;
			continue;
		}
		hid_t signal_dataset_id = 0;
		if(old_format){
			signal_dataset_id = H5Dopen(file_id, (CONCAT3("/Raw/Reads/",read_subgroup_name,"/Signal")).c_str(), H5P_DEFAULT);
		}
		else{
			signal_dataset_id = H5Dopen(file_id, (CONCAT3("/",read_subgroup_name,"/Raw/Signal")).c_str(), H5P_DEFAULT);
		}
		if(signal_dataset_id < 0){
			std::cerr << "Skipping " << read_subgroup_name << " Signal, H5DOpen failed" << std::endl;
			continue;
		}
		hid_t signal_dataspace_id = H5Dget_space(signal_dataset_id);
		if(signal_dataspace_id < 0){
			std::cerr << "Skipping " << read_subgroup_name << " Signal, cannot get the data space" << std::endl;
			continue;
		}
		const hsize_t read_length = H5Sget_simple_extent_npoints(signal_dataspace_id);
		if(read_length < 1){
			std::cerr << "Skipping " << read_subgroup_name << " with reported Signal length " <<  read_length << std::endl;
			continue;
		}
		hid_t memSpace = H5Screate_simple(1, &read_length, NULL);
		if(memSpace < 0){
			std::cerr << "Failed to create a simple memory space specification in the HDF5 API, please report to the software author(s)." << std::endl;
			exit(FAST5_HDF5_API_ERROR);
		}
		short *sequence_buffer = (short *) std::malloc(sizeof(short)*read_length);
		if(H5Dread(signal_dataset_id, H5T_STD_I16LE, memSpace, signal_dataspace_id, H5P_DEFAULT, sequence_buffer) < 0){
			std::cerr << "Skipping " << read_subgroup_name << ", could not get " << read_length << " Signal from bulk FAST5 (HDF5) file '" << fast5_file_name << "'" << std::endl;
			exit(5);
			continue;
		}
		T *t_seq = 0;
		cudaMallocManaged(&t_seq, sizeof(T)*read_length);  CUERR("Cannot allocate CPU memory for FAST5 signal");
		// Convert the FAST5 raw shorts to the desired datatype from the template
		for(int j = 0; j < read_length; j++){
			t_seq[j] = (T) sequence_buffer[j];
		}
		free(sequence_buffer);
		sequences[i] = t_seq;
		sequence_lengths[i] = read_length;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], name_size); CUERR("Cannot allocate CPU memory for reading sequence name from FAST5 file");
		memcpy(sequence_names[local_seq_count_so_far], read_subgroup_name, name_size);

		H5Dclose(signal_dataset_id);
		local_seq_count_so_far++;
	}
	if(read_subgroup_name != NULL) free(read_subgroup_name);
	H5Gclose(read_group);
	H5Fclose(file_id);

	return local_seq_count_so_far;
}

// Function that reads in data from a bulk fast5 file and stores the sequences into a buffer as well as their names.
// bulk5_file_name - the file we are reading the data from
// sequences - the buffer that will store the sequence data
// sequence_names - the names of each sequence stored in the sequence buffer
// sequence_lengths - the lengths of all sequences stored in the sequence buffer
// channel_name - the name of the channel we are reading data from
// start - the start position in the channel to read data from
// end - the end position in the channel to read data from
// returns the number of sequences read in
template <typename T>
int
read_bulk5_data(const char *bulk5_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths, char* channel_name, long long start, long long end){
	/* Open an existing file. */
	hid_t file_id = H5Fopen(bulk5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){
		std::cerr << "Could not open HDF5 file " << bulk5_file_name << " so skipping" << std::endl;
		return 0;
	}	
	
	long long num_samples = end - start;
	
	hid_t signal_dataset_id = H5Dopen(file_id, (CONCAT3("/Raw/", channel_name, "/Signal")).c_str(), H5P_DEFAULT);
	if(signal_dataset_id < 0){
		std::cerr << "Could not get " << (CONCAT3("/Raw/", channel_name, "/Signal")).c_str() << " from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
		H5Fclose(file_id);
		return 0;
	}
	
	// Dataset we want differs based on if we want raw data or non-raw data
	hid_t signal_dataspace_id = H5Dget_space(signal_dataset_id);
	hid_t num_samples_in_raw = H5Sget_simple_extent_npoints(signal_dataspace_id);
	if(num_samples_in_raw < 1){
		std::cerr << "Could not get dimensions of /Raw/" << channel_name << "/Signal dataspace from bulk FAST5 (HDF5) file " << bulk5_file_name << " , aborting" << std::endl;
		H5Fclose(file_id);
		H5Dclose(signal_dataset_id);
		return 0;
	}
	
	hid_t dspace = H5Dget_space(signal_dataset_id);
	const int ndims = H5Sget_simple_extent_ndims(dspace);
	
	hsize_t* dims = new hsize_t[ndims];
	H5Sget_simple_extent_dims(dspace, dims, NULL);
	
	short* samples = (short*) std::malloc(sizeof(short)*num_samples);
	
	hsize_t* start_array = (hsize_t*)malloc(sizeof(hsize_t)*1);
	start_array[0] = start;   // Assuming one dimension array
	hsize_t* count_array = (hsize_t*)malloc(sizeof(hsize_t)*1);
	count_array[0] = num_samples;
	
	if(H5Sselect_hyperslab(signal_dataspace_id, H5S_SELECT_SET, start_array, NULL, count_array, NULL) < 0){
		std::cerr << "Could not get hyperslab from bulk FAST5 (HDF5) file " << bulk5_file_name << " , aborting" << std::endl;
		H5Fclose(file_id);
		H5Dclose(signal_dataset_id);
		free(start_array);
		start_array = NULL;
		free(count_array);
		count_array = NULL;
		return 0;
	}
	
	hsize_t* size_of_elements_needed = (hsize_t*)malloc(sizeof(hsize_t)*1);
	size_of_elements_needed[0] = num_samples;
	hid_t memSpace = H5Screate_simple(ndims, size_of_elements_needed, NULL);
	
	hsize_t* offset_out = (hsize_t*)malloc(sizeof(hsize_t)*1);
	offset_out[0] = 0;   // Assuming one dimension array
	
	hsize_t* count_out = (hsize_t*)malloc(sizeof(hsize_t)*1);
	count_out[0] = num_samples;
	
	if(H5Sselect_hyperslab(memSpace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL) < 0){
		std::cerr << "Could not get hyperslab from bulk FAST5 (HDF5) file " << bulk5_file_name << " , aborting" << std::endl;
		H5Fclose(file_id);
		H5Dclose(signal_dataset_id);
		free(start_array);
		start_array = NULL;
		free(count_array);
		count_array = NULL;
		free(offset_out);
		offset_out = NULL;
		free(count_out);
		count_out = NULL;
		return 0;
	}
	
	if(H5Dread(signal_dataset_id, H5T_STD_I16LE, memSpace, signal_dataspace_id, H5P_DEFAULT, samples) < 0){
		std::cerr << "Could not get /Raw sample data from " << channel_name << " in FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
		H5Fclose(file_id);
		H5Dclose(signal_dataset_id);
		free(start_array);
		start_array = NULL;
		free(count_array);
		count_array = NULL;
		free(offset_out);
		offset_out = NULL;
		free(count_out);
		count_out = NULL;
		return 0;
	}
	
	H5Dclose(signal_dataset_id);
	free(start_array);
	start_array = NULL;
	free(count_array);
	count_array = NULL;
	free(offset_out);
	offset_out = NULL;
	free(count_out);
	count_out = NULL;
	
	T *t_seq = 0;
	cudaMallocHost(&t_seq, sizeof(T)*num_samples);  CUERR("Cannot allocate CPU memory for FAST5 signal");
	// Convert the FAST5 raw shorts to the desired datatype from the template
	for(int j = 0; j < num_samples; j++){
		t_seq[j] = (T) samples[j];
	}
	free(samples);
	sequences[0] = t_seq;
	sequence_lengths[0] = num_samples;
	std::string read_subgroup_name = CONCAT3(CONCAT3(channel_name, "_", std::to_string(start)), "_", std::to_string(end));
	cudaMallocHost(&sequence_names[0], read_subgroup_name.length()); CUERR("Cannot allocate CPU memory for reading sequence name from FAST5 file");
	memcpy(sequence_names[0], read_subgroup_name.c_str(), read_subgroup_name.length());
		
	H5Fclose(file_id);
	
	return 1;
}
#endif

// Function that reads in data from a binary file and stores that data into a buffer
// binary_file_name - the binary file we are reading from
// output_vals - the buffer that will store the data from the file
// num_output_vals - the number of values stored in the buffer
// returns 1 on a success, 0 otherwise
template <typename T>
int
read_binary_data(const char *binary_file_name, T **output_vals, size_t *num_output_vals){

	// See how big the file is, so we can allocate the appropriate buffer
	std::ifstream ifs(binary_file_name, std::ios::binary);
	std::streampos n;
	if(ifs){
		ifs.seekg(0, ifs.end);
		n = ifs.tellg();
		*num_output_vals = n/sizeof(T);
	}
	else{
		std::cerr << "Error reading in file " << binary_file_name << " exiting" << std::endl;
		return 0;
	}
	if((*num_output_vals) == 0){
		std::cerr << binary_file_name << " is empty. Exiting" << std::endl;
		return 0;
	}

	T *out = 0;
	cudaMallocHost(&out, sizeof(T)*n); CUERR("Cannot allocate CPU memory for reading sequence from file");

	ifs.seekg(0, std::ios::beg);
	ifs.read((char *) out, n);

	// Only set the output if all the data was succesfully read in.
	*output_vals = out;
	return 1;
}

// Function that reads in a tsv file and stores the data in a buffer
// text_file_name - the tsv file we will read in
// sequences - the buffer we will be storing the data into
// sequence_names - the names of the sequences stored in the sequence buffer
// sequence_lengths - the lengths of the sequences stored in the sequences buffer
// returns the number of sequences read in
template <typename T>
int
read_tsv_data(const char *text_file_name, T **sequences, char **sequence_names, size_t *sequence_lengths){
	int local_seq_count_so_far = 0;
	// One sequence per line, values tab separated.
	
	std::ifstream ifs(text_file_name);
	if(!ifs){
		std::cerr << "Error reading in file " << text_file_name << " exiting" << std::endl;
		return 0;
	}
	for(std::string line; std::getline(ifs, line); ){
		// Assumption is that the first value is the sequence name (or in the case of the UCR time series archive, the sequence class identifier).
		int numDataColumns = std::count(line.begin(), line.end(), '\t');
		sequence_lengths[local_seq_count_so_far] = numDataColumns;
		T *this_seq;
		cudaMallocHost(&this_seq, sizeof(T)*numDataColumns); CUERR("Cannot allocate CPU memory for reading sequence from TSV file");
		sequences[local_seq_count_so_far] = this_seq;

		std::istringstream iss(line);
		std::string seq_name;
		iss >> seq_name;
		cudaMallocHost(&sequence_names[local_seq_count_so_far], seq_name.length()+1); CUERR("Cannot allocate CPU memory for reading sequence name from TSV file");
		memcpy(sequence_names[local_seq_count_so_far], seq_name.c_str(), seq_name.length()+1);
		int element_count = 0;
		while(iss.good()){
			iss >> this_seq[element_count++]; // automatically does string -> numeric value conversion
		}
		local_seq_count_so_far++;
	}
	return local_seq_count_so_far;
}

//-------------------------
//-------------------------
//-----READ SEQUENCES------
//-------------------------
//-------------------------

// Function that reads in multiple text files and stores their information into a buffer
// filenames - a list of text files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// returns the total number of sequences read in
template<typename T>
int readSequenceTextFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){
	cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*num_files); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " text data file" : " text data files") << ", total sequence count " << num_files << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	for(int i = 0; i < num_files; ++i){
		int newDotTotal = 100*((float) i/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[i%4];
		}

		if(!read_text_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_lengths) + actual_count)){
		std::cerr << "Error reading in file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}
		(*sequence_names)[i] = filenames[i];
	}
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

#if HDF5_SUPPORTED == 1

// Function that reads in multiple fast5 files and stores their information into a buffer
// filenames - a list of fast5 files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// returns the total number of sequences read in
template<typename T>
int readSequenceFAST5Files(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){

	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " FAST5 data file" : " FAST5 data files");
	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	for(int i = 0; i < num_files; ++i){
		size_t seq_count_this_file = 0;
		scan_fast5_data(filenames[i], &seq_count_this_file);
		total_seq_count += seq_count_this_file;
	}
	if(total_seq_count == 0){
		std::cerr << "No sequences found in Fast5 file. Exiting." << std::endl;
		return 0;
	}
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	for(int i = 0; i < num_files; ++i){
		int newDotTotal = 100*((float) i/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[i%4];
		}

		size_t num_seqs_this_file = read_fast5_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count);
		if(num_seqs_this_file < 1){
			std::cerr << "Error reading in FAST5 file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count += num_seqs_this_file;
		}
	}
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

// Function that reads in multiple bulk fast5 files and stores their information into a buffer
// filenames - a list of the bulk fast5 files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// returns the total number of sequences read in
template<typename T>
int readSequenceBULK5Files(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths, int instrand){

	std::cerr << "Step 1 of 3: Loading BULK5 data file";
	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	std::vector< std::vector< std::pair<char*, std::pair <long long,long long> > > > all_channel_ranges;
	for(int i = 0; i < num_files; i++){
		size_t seq_count_here = 0;
		std::vector< std::pair<char*, std::pair <long long,long long> > > channel_ranges;
		scan_bulk5_data(filenames[i], channel_ranges, &seq_count_here, instrand);
		total_seq_count += seq_count_here;
		all_channel_ranges.push_back(channel_ranges);
	}
		
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	
	// std::vector< std::map<char*, std::vector< std::pair <long long,long long> > > >::iterator channel_ranges = all_channel_ranges.begin();
	
	for(std::vector< std::vector< std::pair< char*, std::pair <long long,long long> > > >::size_type i = 0; i != all_channel_ranges.size(); i++){
		std::vector< std::pair< char*, std::pair <long long,long long> > > channel_ranges = all_channel_ranges[i];
		int file_pos = 0;
		int newDotTotal = 100*((float) file_pos/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[file_pos%4];
		}
		// std::map<char*, std::vector< std::pair <long long,long long> > >::iterator channel_it = channel_ranges.begin();
		// Itterate over all channels using their start and end points to optain data from
		// while(channel_it != channel_ranges.end()){
		// for(std::vector< std::pair< char*, std::pair <long long,long long> > >::size_type j = 0; j != channel_ranges.size(); j++){
		for(std::vector< std::pair< char*, std::pair <long long,long long> > >::iterator channel_name_range = channel_ranges.begin(); channel_name_range != channel_ranges.end(); channel_name_range++){
			
			// std::pair channel_name_range = channel_ranges[j];
			
			char * channel_name = channel_name_range->first;
			
			long long start = channel_name_range->second.first;
			long long end = channel_name_range->second.second;

			size_t num_seqs_this_file = read_bulk5_data<T>(filenames[file_pos], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count, channel_name, start, end);
			if(num_seqs_this_file < 1){
				std::cerr << "Error reading in BULK5 file " << filenames[file_pos] << ", exiting" << std::endl;
				return 0;
			}
			else{
				actual_count += num_seqs_this_file;
			}
		}
		file_pos++;
	}
	
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}
#endif

// Function that reads in multiple fasta files and stores their information into a buffer
// filenames - a list of fasta files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// rna - flag that determines if we are reading in rna data or dna data
// signal_type - the type of signal being read in
// strand_flags - flag that determines the strand to read in
// returns the total number of sequences read in
template<typename T>
int readSequenceFASTAFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths, int rna, short signal_type, short strand_flags){
	
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " FASTA data file" : " FASTA data files");
	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	for(int i = 0; i < num_files; ++i){
		size_t seq_count_this_file = 0;
		scan_fasta_data(filenames[i], &seq_count_this_file);
		total_seq_count += seq_count_this_file;
	}
	if(total_seq_count == 0){
		std::cerr << "No sequences found in FastA file. Exiting." << std::endl;
		return 0;
	}
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	int sequence_position = 0;
	for(int i = 0; i < num_files; ++i){
		int newDotTotal = 100*((float) i/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[i%4];
		}
		std::ifstream f(filenames[i]);
		f.seekg(0, f.end);
		int input_buffer_size = f.tellg();
		f.seekg(0, f.beg);
		char *input = (char *) std::malloc(input_buffer_size*sizeof(char));
		
		std::string line;
		long input_length = 0;
		long num_seqs_this_file = 0;
		while (!f.eof()) {
			std::getline(f,line);
			if (line.length() == 0) // blank or header line
				continue;
			else if(line[0] == '>'){
				if(input_length != 0){
					short* tmp_vals = 0;
					long num_seqs_this_input = 0;
					// std::cerr << input << std::endl;
					if(rna){
						tmp_vals = convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_input);	
					} else{
						tmp_vals = convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_input);
					}
					(*sequences)[sequence_position] = shortToTemplate<T>(tmp_vals, (long long)num_seqs_this_input);
					(*sequence_lengths)[sequence_position] = (size_t)num_seqs_this_input;
					actual_count += (size_t)num_seqs_this_input;
					num_seqs_this_file += (size_t)num_seqs_this_input;
					input_length = 0;
					sequence_position++;
				}
				int name_size = line.size() + 1;
				char* seq_name_c = (char *) std::malloc(name_size);
				std::strcpy(seq_name_c, line.c_str());
				
				// if(sequence_position == 0){
				cudaMallocHost(&((*sequence_names)[sequence_position]), name_size); CUERR("Cannot allocate CPU memory for reading sequence name from FASTA file");
				memcpy((*sequence_names)[sequence_position], seq_name_c, name_size);
				// std::cerr << (*sequence_names)[sequence_position] << std::endl;
				// }
				free(seq_name_c);
			}
			else{
				// Note that the following only works for ASCII, which is what a sane FastA file is encoded as.
				std::transform(line.begin(), line.end(), line.begin(), [] (char c) {return (std::toupper(c));});
				strncpy(&input[input_length], line.c_str(), line.length());
				input_length += line.length();
			}
		}
		f.close();
		if(num_seqs_this_file == 0){
			std::cerr << "Error reading in FAST5 file " << filenames[i] << ", skipping" << std::endl;
			continue;
		}
		num_seqs_this_file = 0;
		if(rna){
			(*sequences)[sequence_position] = shortToTemplate<T>(convert_rna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_file), (long long)num_seqs_this_file);
		} else{
			(*sequences)[sequence_position] = shortToTemplate<T>(convert_dna_to_shorts(input, input_length, signal_type, strand_flags, &num_seqs_this_file), (long long)num_seqs_this_file);
		}
		(*sequence_lengths)[sequence_position] = (size_t)num_seqs_this_file;
		sequence_position++;
	}
	if(actual_count == 0){
		std::cerr << "No data was read from any FAST5 files." << std::endl;
		return 0;
	}
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return sequence_position;
	
}

// Function that reads in multiple binary files and stores their information into a buffer
// filenames - a list of the binary files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// returns the total number of sequences read in
template<typename T>
int readSequenceBinaryFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){
	cudaMallocHost(sequences, sizeof(T *)*num_files); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*num_files); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*num_files); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " binary data file" : " binary data files") << ", total sequence count " << num_files << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	for(int i = 0; i < num_files; ++i){
		int newDotTotal = 100*((float) i/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[i%4];
		}

		if(!read_binary_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_lengths) + actual_count)){
			std::cerr << "Error reading in file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count++;
		}
		(*sequence_names)[i] = filenames[i];
	}
	if(actual_count == 0){
		std::cerr << "No data was read from any binary files." << std::endl;
		return 0;
	}
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

// Function that reads in multiple tsv files and stores their information into a buffer
// filenames - a list of the tsv files we will be reading in
// num_files - the number of files in the filenames list
// sequences - the buffer that will store all data from the list of files
// sequence_names - the names of all sequences stored in the sequences buffer
// sequence_lengths - the lengths of all sequences stored in the sequences buffer
// returns the total number of sequences read in
template<typename T>
int readSequenceTSVFiles(char **filenames, int num_files, T ***sequences, char ***sequence_names, size_t **sequence_lengths){

	std::cerr << "Step 1 of 3: Loading " << num_files << (num_files == 1 ? " TSV data file" : " TSV data files");

	// Need two passes: 1st figure out how many sequences there are, then in the 2nd we read the sequences into memory.
	size_t total_seq_count = 0;
	for(int i = 0; i < num_files; ++i){
		size_t seq_count_this_file = 0;
		scan_tsv_data(filenames[i], &seq_count_this_file);
		total_seq_count += seq_count_this_file;
	}
	std::cerr << ", total sequence count " << total_seq_count << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	cudaMallocHost(sequences, sizeof(T *)*total_seq_count); CUERR("Allocating CPU memory for sequence pointers");
	cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");
	cudaMallocHost(sequence_lengths, sizeof(size_t)*total_seq_count); CUERR("Allocating CPU memory for sequence lengths");

	int dotsPrinted = 0;
	char spinner[4] = { '|', '/', '-', '\\'};
	int actual_count = 0;
	for(int i = 0; i < num_files; ++i){
		int newDotTotal = 100*((float) i/(num_files-1));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[i%4];
		}

		size_t num_seqs_this_file = read_tsv_data<T>(filenames[i], (*sequences) + actual_count, (*sequence_names) + actual_count, (*sequence_lengths) + actual_count);
		if(num_seqs_this_file < 1){
			std::cerr << "Error reading in TSV file " << filenames[i] << ", skipping" << std::endl;
		}
		else{
			actual_count += num_seqs_this_file;
		}
	}
	if(dotsPrinted < 100){while(dotsPrinted++ < 99){std::cerr << ".";} std::cerr << "|";}
	std::cerr << std::endl;
	return actual_count;
}

// Function that takes a 2D buffer and merges all data into a single buffer
// input_vals - the 2D buffer we will be merging
// num_input_vals - the number of values in each sequence in input_vals
// subject_offsets - a vector that will be storing the start points as well as the names of each sequence stored in the merged buffer
// sequence_names - the sequence names of the sequences stored in the 2D buffer
// num_seq - the number of sequences in input_vals
// total_vals - a buffer that will contain the size of the merged buffer
// returns the merged buffer
template <class T>
T*
merge_data(T **input_vals, size_t *num_input_vals, std::vector< std::pair<size_t, char*> >& subject_offsets, char** sequence_names, int num_seq, size_t *total_vals){
	*total_vals = 0;
	for(int i = 0; i < num_seq; i++){
		*total_vals += num_input_vals[i];
	}
	if(*total_vals == 0){
		std::cerr << "No data given. Exiting." << std::endl;
		return 0;
	}
	
	T* all_values = 0; 
	cudaMallocHost(&all_values, sizeof(T)*(*total_vals)); CUERR("Allocating CPU memory for all values");
	
	size_t total_vals_copied = 0;
	for(int i = 0; i < num_seq; i++){
		size_t tmp_val = num_input_vals[i];
		if(tmp_val == 0){
			continue;
		}
		cudaMemcpy(&all_values[total_vals_copied], input_vals[i], tmp_val*sizeof(T), cudaMemcpyHostToHost);
		total_vals_copied += tmp_val;
		cudaFreeHost(input_vals[i]);		CUERR("Free CPU input vals");
		subject_offsets.push_back(std::pair<size_t, char*>(total_vals_copied, sequence_names[i]));
	}
	cudaFreeHost(input_vals);		CUERR("Free CPU all values");
	cudaFreeHost(num_input_vals);	CUERR("Free CPU num values");
	return all_values;
}

// Function that gets a list of filenames and reads in the data from each of them
// filenames - the list of filenames we will be reading in
// num_files - the number of files in filenames
// output_vals - the buffer that will contain all data from the files
// sequence_names - the buffer that will contain all the sequence names from the files
// num_output_vals - the buffer that will contain the number of values in each sequence read in from the files
// rna - flag that determines if we are reading in rna or dna data from a fasta file
// signal_type - the type of signal being read in from a fasta file
// strand_flags - flag that determines the strand to read in from a fasta file
// returns the total number of sequences read in from the files
// TODO: add compatible files (bin and tsv)
template <class T>
int
read_data(char **filenames, int num_files, T ***output_vals, char ***sequence_names, size_t **num_output_vals, int instrand, int rna, short signal_type, short strand_flags){
	// STRAND = instrand;
	int total_count = 0;
	
	size_t total_seq_count = 0;
	
	for(int i = 0; i < num_files; ++i){
		if(!checkEnding(std::string(filenames[i]))){
			continue;
		}
		size_t seq_count_this_file = 0;
		char tmp_filename[256];
		strcpy(tmp_filename, filenames[i]);
		char *extension;
		char *ptr = strtok(tmp_filename, ".");
		while(ptr = strtok(NULL, ".")) {
			extension = ptr;
		}
		if(strcmp(extension, "txt") == 0){
			seq_count_this_file = 1;
		} 
		
		#if HDF5_SUPPORTED == 1
			else if(strcmp(extension, "fast5") == 0) {
				if(isBulkFast5(filenames[i])){
					std::vector< std::pair<char*, std::pair <long long,long long> > > channel_ranges;
					scan_bulk5_data(filenames[i], channel_ranges, &seq_count_this_file, instrand);
				} else{
					scan_fast5_data(filenames[i], &seq_count_this_file);
				}
			} 
		#endif
		
		else if(strcmp(extension, "fna") == 0){
			scan_fasta_data(filenames[i], &seq_count_this_file);
		} else{
			seq_count_this_file = 1;
		}
		total_seq_count += seq_count_this_file;
	}
	if(total_seq_count == 0){
		std::cerr << "No valid files found. Exiting." << std::endl;
		return 0;
	}
	
	cudaMallocHost(output_vals, sizeof(T *)*total_seq_count);				CUERR("Allocating cpu memory for values");
	cudaMallocHost(sequence_names, sizeof(char *)*total_seq_count);       CUERR("Allocating cpu memory for names");
	cudaMallocHost(num_output_vals, sizeof(size_t)*total_seq_count);      CUERR("Allocating cpu memory for num values");
	for(int i = 0; i < num_files; i++){
		if(!checkEnding(std::string(filenames[i]))){
			continue;
		}
		char tmp_filename[256];
		strcpy(tmp_filename, filenames[i]);
		char *extension;
		char *ptr = strtok(tmp_filename, ".");
		while(ptr = strtok(NULL, ".")) {
			extension = ptr;
		}
		
		T** tmp_out_vals;
		char** tmp_names;
		size_t* tmp_num_vals;
		size_t tmp_count = 0;
		// std::cerr << "extension: " << extension << std::endl;
		if(strcmp(extension, "txt") == 0){
			tmp_count = readSequenceTextFiles(&filenames[i], 1, &tmp_out_vals, &tmp_names, &tmp_num_vals);
		} 
		
		#if HDF5_SUPPORTED == 1
			else if(strcmp(extension, "fast5") == 0){
				if(isBulkFast5(filenames[i])){
					tmp_count = readSequenceBULK5Files(&filenames[i], 1, &tmp_out_vals, &tmp_names, &tmp_num_vals, instrand);
				} else{
					tmp_count = readSequenceFAST5Files(&filenames[i], 1, &tmp_out_vals, &tmp_names, &tmp_num_vals);
				}
			} 
		#endif
		
		else if(strcmp(extension, "fna") == 0){
			tmp_count = readSequenceFASTAFiles(&filenames[i], 1, &tmp_out_vals, &tmp_names, &tmp_num_vals, rna, signal_type, strand_flags);
		} else{
			tmp_count = readSequenceBinaryFiles(&filenames[i], 1, &tmp_out_vals, &tmp_names, &tmp_num_vals);
		}
		for(int j = 0; j < tmp_count; j++){
			(*output_vals)[total_count+j] = tmp_out_vals[j];
			(*sequence_names)[total_count+j] = tmp_names[j];
			(*num_output_vals)[total_count+j] = tmp_num_vals[j];
		}
		total_count += tmp_count;
		std::cerr << "total count: " << total_count << std::endl;
	}
	
	return total_count;
}

#endif
