#ifndef __ALL_UTILLS_H
#define __ALL_UTILLS_H

#include "exit_codes.hpp"

#if defined(_WIN32)
	#define end_slash "\\"
#else
	#define end_slash "/"
#endif

#define SSTR( x ) dynamic_cast< std::ostringstream & >(  ( std::ostringstream() << std::dec << x ) ).str()

#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <cerrno>
#include <dirent.h>

#include "IntervalTree.h"
// #include "strand_utils.hpp"

#if HDF5_SUPPORTED == 1
extern "C"{
	#include "hdf5.h"
}
#endif

typedef IntervalTree<size_t, std::string> ITree;
	
#define CONCAT2(a,b)	 (std::string(a)+std::string(b))
#define CONCAT3(a,b,c)	 (std::string(a)+std::string(b)+std::string(c))

// extern int STRAND;

// Function that converts a string to char pointer
// tmp_string - the string we want to convert_dna_to_shorts
// returns converted char pointer
char* stringToChar(std::string tmp_string){
	char* cstr = (char*) malloc(tmp_string.size() + 1);
	strcpy(cstr, tmp_string.c_str());
	return cstr;
}


// Function that gets the number of values in a file
// is - the input stream for the file
// buff - the buffer that will store the contents of the file
// returns the number of values read in
inline unsigned int FileRead( std::istream & is, std::vector <char> & buff ) {
	is.read( &buff[0], buff.size() );
	return is.gcount();
}

// Function that counts the number of lines in a buffer
// buff - the buffer to be checked
// sz - the size of the buffer
// returns the number of new lines in the buffer
inline unsigned int CountLines( const std::vector <char> & buff, int sz ) {
	int newlines = 0;
	const char * p = &buff[0];
	for ( int i = 0; i < sz; i++ ) {
		if ( p[i] == '\n' ) {
			newlines++;
		}
	}
	return newlines;
}

// Function that converts a short to the given template value
// data - the short buffer to be converted
// data_length - the length of the buffer passed in
// returns a new buffer that is of type template with the short data stored in it
template <class T>
T* shortToTemplate(short* data, long long data_length){
	T* return_data;
	return_data = (T*)malloc(sizeof(T)*data_length);
	std::transform(data, data + data_length, return_data, [](short s){ return (T)s; });
	return return_data;
}

// From https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
// Checks if a string has a specific ending
// fullString - the string we want to check
// ending - the ending we are looking for
// returns true if found, false if not
inline bool hasEnding (std::string const &fullString, std::string const &ending){
	if(fullString.length() >= ending.length()){
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

#if HDF5_SUPPORTED == 1
// Function that checks if a file is a Bulk5 file
// file_path - the file we want to check
// returns true if a Bulk5 file, false otherwise
bool isBulkFast5(char* file_path){
	/* Open an existing file. */
	hid_t file_id = H5Fopen(file_path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){
		std::cerr << "Could not open HDF5 file " << file_path << " so skipping" << std::endl;
		return 0;
	}
	// bool result = false;
	hid_t read_group = 0;
	H5E_BEGIN_TRY {
		read_group = H5Gopen(file_id, "/Raw/Channel_1", H5P_DEFAULT);
	} H5E_END_TRY
	bool result = read_group > 0;
	if(result) H5Gclose(read_group);
	H5Fclose(file_id);
	// if(H5Lexists(file_id, "/Raw/Channel_1", H5P_DEFAULT) != 0)
		// result = true;
	// H5Fclose(file_id);
	return result;
}
#endif

const int num_accepted_extensions = 5;
const std::string extensions[num_accepted_extensions] = {".fast5", ".txt", ".fna", ".bin", ".tsv"};
// Function that checks the ending of a file and determines if it is a valid file
// fullString - the file that will be checked
// returns true if the file extension is found within the extensions array. Otherwise returns false.
inline bool checkEnding(std::string const &fullString){
	for(int i = 0; i < num_accepted_extensions; i++){
		if(fullString.length() >= extensions[i].length() && 0 == fullString.compare(fullString.length() - extensions[i].length(), extensions[i].length(), extensions[i])){
			return true;
		}
	}
	return false;
}

// Function that gets all valid files in a directory and stores their file path into a vector
// directory_path - path of the directory that contains all valid files
// all_files - char array that will be populated with the full paths of all valid files
// returns number of compatible files if successful, 0 otherwise
inline int getAllFilesFromDir(char* directory_path, char*** all_files){
	// Code found and refactored from https://www.geeksforgeeks.org/c-program-list-files-sub-directories-directory/
	std::string dir_path = directory_path;
	struct dirent *de;  // Pointer for directory entry

	if(dir_path.back() != '\\' && dir_path.back() != '/'){
		dir_path = dir_path + end_slash;
	}

	// opendir() returns a pointer of DIR type.
	DIR *dr = opendir(directory_path);

	if (dr == NULL){  // opendir returns NULL if couldn't open directory
		std::cerr << "Could not open directory: " << directory_path << ". Exiting" << std::endl;
		return 0;
	}

	int num_compatible_files = 0;
	// Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
	// for readdir()
	while ((de = readdir(dr)) != NULL){
		std::string file_path = dir_path + de->d_name;
		if(checkEnding(file_path)){
			num_compatible_files++;
		}
	}

	if(num_compatible_files == 0){
		std::cerr << "No compatible files could be found in directory " << directory_path << " Please make sure you're passing in a directory that contains at least one compatible file. Exiting." << std::endl;
		closedir(dr);
		return 0;
	}
	
	*all_files = (char**)malloc(sizeof(char*)*num_compatible_files);
	num_compatible_files = 0;
	rewinddir(dr);
	while ((de = readdir(dr)) != NULL){
		std::string file_path = dir_path + de->d_name;
		if(checkEnding(file_path)){
			char* cfile_path = (char*)malloc(sizeof(char)*file_path.size()+1);
			strcpy(cfile_path, file_path.c_str());
			(*all_files)[num_compatible_files] = cfile_path;
			num_compatible_files++;
		}
	}
	closedir(dr);
	return num_compatible_files;
}

// Function that populates an interval tree with the contents of a BED file
// bed_intervals - the interval tree that we are going to populate
// bed_filename - the BED file that we will be reading values from
// verbose - verbose mode
// returns 1 on success, 0 otherwise
inline int populateITree(ITree::interval_vector& bed_intervals, char* bed_filename, int verbose){
	if(verbose) std::cerr << "Populating interval tree with contents of BED file" << std::endl;
	errno = 0;
	std::ifstream bfile(bed_filename, std::ios::binary);
	if(errno || !bfile.is_open()){
		std::cerr << "Could not open BED file  " << bed_filename << " for reading, aborting: " << strerror(errno) << std::endl;
		return 0;
	}

	std::string strCRLF = "\r\n";
	std::string strENDL = "\n";
	std::string strCRL = "\r";

	std::string line;
	bool data_read = false;
	while(bfile.good()){
		// Chedk for tabs separating information in the BED file
		std::getline(bfile, line);
		if(line.length() == 0)
			continue;

		if(hasEnding(line, strCRLF)){
			line = line.substr(0, line.size()-2);
		} else if(hasEnding(line, strENDL)){
			line = line.substr(0, line.size()-1);
		} else if(hasEnding(line, strCRL)){
			line = line.substr(0, line.size()-1);	// Removes the /r that could be present if file was generated on windows
		}

		size_t tab_pos = line.find_first_of('\t');
		if(tab_pos == std::string::npos){
			std::cerr << "BED file " << bed_filename << " is malformatted, aborting. Expect tab in line '" << line << "'" << std::endl;
			bfile.close();
			return 0;
		}

		std::string delimiter = "\t";
		std::istringstream ss(line);
		std::vector<std::string> bed_data;
		std::string substr;

		while(std::getline(ss, substr, '\t')){
			bed_data.push_back(substr);
		}
		if(bed_data.size() < 3) {
			std::cerr << "Not enough values were found on line (" << line << ") so exiting." << std::endl;
			return 0;
		}

		// We only care about the first three fields in the file (https://genome.ucsc.edu/FAQ/FAQformat.html):
		// chrom - The name of the chromosome (e.g. chr3, chrY, chr2_random) or scaffold (e.g. scaffold10671).
		// chromStart - The starting position of the feature in the chromosome or scaffold. The first base in a chromosome is numbered 0.
		// chromEnd - The ending position of the feature in the chromosome or scaffold. The chromEnd base is not included in the display of the feature.
		std::string chrom_name = bed_data[0];
		std::string chrom_start = bed_data[1];
		std::string chrom_end = bed_data[2];

		std::stringstream sstream_start(chrom_start);
		std::stringstream sstream_end(chrom_end);

		size_t chrom_start_i;
		size_t chrom_end_i;

		if(!(sstream_start >> chrom_start_i)) {
			std::cerr << "Start value for chromosome is invalid (" << chrom_start << "). Exiting." << std::endl;
			return 0;
		}
		if(!(sstream_end >> chrom_end_i)){
			std::cerr << "End value for chromosome is invalid (" << chrom_end << "). Exiting." << std::endl;
			return 0;
		}
		if(verbose) std::cerr << "Adding to vector: " << chrom_name << " " << chrom_start_i << " " << chrom_end_i << std::endl;
		bed_intervals.push_back(Interval<size_t, std::string>(chrom_start_i, (chrom_end_i-1), chrom_name));	// Subtract 1 from chrom_end since it is not included
		data_read = true;
	}
	bfile.close();
	if(!data_read){
		std::cerr << "BED file was empty, so no data was read in. Please make sure you are providing a BED file with data in it and rerun the program. Exiting." << std::endl;
		return 0;
	}
	return 1;
}

//-------------------------
//-------------------------
//-------SCAN DATA---------
//-------------------------
//-------------------------

#if HDF5_SUPPORTED == 1
// Function that scans a fast5 file and gets the number of sequences it contains
// fast5_file_name - the fast5 file we will be scanning
// num_sequences - a pointer that will store the number of sequences contained in the fast5 file
// returns 1 on success. FAST5_FILE_CONTENTS_UNRECOGNIZED otherwise
int
scan_fast5_data(const char *fast5_file_name, size_t *num_sequences){
	hsize_t num_read_objects;

	/* Open an existing file. */
	hid_t file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){
		std::cerr << "Could not open HDF5 file " << fast5_file_name << std::endl;
		return FAST5_FILE_UNREADABLE;
	}
	H5Eset_auto1(NULL, NULL);
	// Old format, one read per file
	hid_t read_group = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
	if(read_group < 0){ // New formst, multiple reads per file
		read_group = H5Gopen(file_id, "/", H5P_DEFAULT);
	}

	if(read_group < 0 || H5Gget_num_objs(read_group, &num_read_objects)){
		std::cerr << "Could not get read groups from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
		H5Fclose(file_id);
		H5Gclose(read_group);
		return FAST5_FILE_CONTENTS_UNRECOGNIZED;
	}
	hsize_t num_read_objects_rejected = 0;
	char read_subgroup_name[6]; // only care about the first five letters for this check, plus null string terminator
	for(int j = 0; j < num_read_objects; ++j){
		// See if the object looks like a read based on its name
		size_t name_size = H5Gget_objname_by_idx(read_group, j, read_subgroup_name, 6);
		// Should have the form Read_# (old) or read_????? (new)
		if(name_size < 5 || (read_subgroup_name[0] != 'R' && read_subgroup_name[0] != 'r') 
						 || read_subgroup_name[1] != 'e' 
						 || read_subgroup_name[2] != 'a' 
						 || read_subgroup_name[3] != 'd' 
						 || read_subgroup_name[4] != '_'){ 
			std::cout << "Skipping unexpected HDF5 object " << read_subgroup_name << std::endl;
			num_read_objects_rejected++;
			continue;
		}
	}
	*num_sequences = (size_t) (num_read_objects - num_read_objects_rejected);
	return 1;
}

// Function that returns a value of a given data point in a Bulk5 file
// channel_dataset_id - the channel dataset to obtain the end value from
// dataspace_index_wanted - the dataspace of the end value
// channel_dataspace_id - the dataspace of the channel
// position - the position in the channel to get the value from
// returns the value
long long getCoordVal(hid_t channel_dataset_id, hid_t dataspace_index_wanted, hid_t channel_dataspace_id, unsigned long position){

	hsize_t* coords = (hsize_t*)malloc(sizeof(hsize_t)*1);
	coords[0] = position;

	if(H5Sselect_elements(channel_dataspace_id, H5S_SELECT_SET, 1, coords) < 0){  // Select the set of positions found from above
		std::cerr << "Unable to select single element, aborting." << std::endl;
		exit(1);
	}

	long long next_end_value;
	hsize_t *size_of_elements_needed = (hsize_t*) std::malloc(sizeof(hsize_t)*1);
	size_of_elements_needed[0] = 1;
	hid_t memSpace = H5Screate_simple(1, size_of_elements_needed, NULL);
	if(H5Dread(channel_dataset_id, dataspace_index_wanted, memSpace, channel_dataspace_id, H5P_DEFAULT, &next_end_value) < 0){  // Read all start and end positions into an array
		std::cerr << "Could not get single element from dataset 'acquisition_raw_index' members from bulk FAST5 (HDF5) file, aborting." << std::endl;
		exit(1);
	}
	free(coords);
	coords = NULL;
	return next_end_value;
}

// Function that scans a bulk fast5 file and gets the number of sequences it contains as well as the ranges in each channel that are considered to be instrand
// TODO: Some bulk fast5 files have different values that represent whether a channel is considered to be instrand or not. We will need to motify this code so that the instrad value is not hardcoded
// bulk5_file_name - the bulk fast5 file we will be scanning
// channel_ranges - a vector that will be populated by the channel names and the ranges in each channel where data was instrand
// total_seqsize - a pointer that will store the number of sequences contained in the bulk fast5 file
// returns 1 on success. 0 on fail
int
scan_bulk5_data(const char *bulk5_file_name, std::vector< std::pair<char*, std::pair <long long,long long> > >& channel_ranges, size_t *total_seqsize, int instrand){

	hid_t file_id = H5Fopen(bulk5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0){
		std::cerr << "Could not open HDF5 file " << bulk5_file_name << ", aborting" << std::endl;
		return 0;
	}

	hid_t state_data_id_group = H5Gopen(file_id, "/StateData", H5P_DEFAULT);
	if(state_data_id_group < 0){
		std::cerr << "Could not get /StateData group from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
		H5Fclose(file_id);
		return 0;
	}

	hsize_t num_channels;
	if(H5Gget_num_objs(state_data_id_group, &num_channels)){
		std::cerr << "Could not get /StateData group objects from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting" << std::endl;
		H5Gclose(state_data_id_group);
		H5Fclose(file_id);
		return 0;
	}
	
	std::cerr << "There are " << num_channels << " channels in the bulk fast5 file." << std::endl;
	
	// Read through all channels once to see how big the biggest sequence is so we can allocate the memory for it
	for(int j = 0; j < num_channels; j++){
		ssize_t name_size = H5Gget_objname_by_idx(state_data_id_group, j, NULL, 0);
		name_size++; // for NULL termination
		char *channel_name = NULL;
		errno = 0;
		channel_name = (char *) std::malloc(name_size);
		if(errno || (channel_name == NULL)){
			std::cerr << "Error in malloc for HDF5 group name: " << strerror(errno) << std::endl;
			exit(1);
		}
		H5Gget_objname_by_idx(state_data_id_group, j, channel_name, (size_t) name_size);
		if(channel_name[0] != 'C' || channel_name[1] != 'h' 
								  || channel_name[2] != 'a' 
								  || channel_name[3] != 'n' 
								  || channel_name[4] != 'n' 
								  || channel_name[5] != 'e' 
								  || channel_name[6] != 'l' 
								  || channel_name[7] != '_'){ // 	should have the form Read_#
			std::cerr << "Skipping unexpected HDF5 object " << state_data_id_group << std::endl;
			free(channel_name);
			channel_name = NULL;
			continue;
		}

		hid_t channel_dataset_id = H5Dopen(file_id, (CONCAT3("/StateData/",channel_name,"/States")).c_str(), H5P_DEFAULT);
		hid_t channel_dataspace_id = H5Dget_space(channel_dataset_id);
		hid_t num_data_from_channel = H5Sget_simple_extent_npoints(channel_dataspace_id);

		if(num_data_from_channel < 1){
			std::cerr << "Could not get dimensions of /StateData/" << channel_name << "//States dataspace from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting. " << strerror(errno) << std::endl;
			H5Dclose(channel_dataset_id);
			H5Gclose(state_data_id_group);
			H5Fclose(file_id);
			free(channel_name);
			channel_name = NULL;
			return 0;
		}

		errno = 0;
		int *channel_states = (int*) std::malloc(sizeof(int)*num_data_from_channel);  // Will contain the states of all channels in the file
		if(errno || (channel_states == NULL)){
			std::cerr << "Error in malloc for HDF5 state data (" << num_data_from_channel << " states): " << strerror(errno) << std::endl;
			H5Dclose(channel_dataset_id);
			H5Gclose(state_data_id_group);
			H5Fclose(file_id);
			free(channel_name);
			channel_name = NULL;
			return 0;
		}

		hid_t dataspace_state_wanted = H5Tcreate(H5T_COMPOUND, sizeof(int));  // Need dataspace for the states of the channels
		if(H5Tinsert(dataspace_state_wanted, "summary_state", 0, H5T_STD_I32LE) < 0){
			std::cerr << "Unable to insert dataset 'summary_state', aborting. " << strerror(errno) << std::endl;
			H5Dclose(channel_dataset_id);
			H5Gclose(state_data_id_group);
			H5Fclose(file_id);
			free(channel_name);
			channel_name = NULL;
			free(channel_states);
			channel_states = NULL;
			return 0;
		}

		hid_t dataspace_index_wanted = H5Tcreate(H5T_COMPOUND, sizeof(unsigned long long));  // Need dataspace for the starting indeces of the channels
		if(H5Tinsert(dataspace_index_wanted, "acquisition_raw_index", 0, H5T_STD_U64LE) < 0){
			std::cerr << "Unable to insert dataset 'acquisition_raw_index', aborting. " << strerror(errno) << std::endl;
			H5Dclose(channel_dataset_id);
			H5Gclose(state_data_id_group);
			H5Fclose(file_id);
			free(channel_name);
			channel_name = NULL;
			free(channel_states);
			channel_states = NULL;
			return 0;
		}

		// Read only the channel states we want
		if(H5Dread(channel_dataset_id, dataspace_state_wanted, H5S_ALL, H5S_ALL, H5P_DEFAULT, channel_states) < 0){
			std::cerr << "Could not get /StateData/" << channel_name << "/States dataset 'summary_state' members from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting. " << strerror(errno) << std::endl;
			H5Dclose(channel_dataset_id);
			H5Gclose(state_data_id_group);
			H5Fclose(file_id);
			free(channel_name);
			channel_name = NULL;
			free(channel_states);
			channel_states = NULL;
			return 0;
		}

		hsize_t *strand_start_ends_pos = (hsize_t*) std::malloc(sizeof(hsize_t)*num_data_from_channel*2); //Max possible number of strands *2 for start and end locations
		hsize_t strand_start_ends_loc = 0;
		bool strand_at_end = false;
		// if(verbose) std::cerr << "Starts and ends for channel " << channel_name << ":" << std::endl;
		for(int count = 0; count < num_data_from_channel; count++){
			if(channel_states[count] == instrand){
				strand_start_ends_pos[strand_start_ends_loc++] = count; // Start
				// if(verbose) std::cerr << "Start: " << count << ", ";
				if(count + 1 != num_data_from_channel){
					strand_start_ends_pos[strand_start_ends_loc++] = count + 1; // End
					// if(verbose) std::cerr << "End: " << count + 1 << std::endl;
				} else{
					strand_at_end = true;
					// strand_start_ends_loc++;
				}
			}
		}

		if(strand_start_ends_loc != 0){
			hsize_t* coords = (hsize_t*) calloc(strand_start_ends_loc, sizeof(hsize_t));
			if(coords == NULL){
				std::cerr << "Could not allocate " << strand_start_ends_loc << " for coords. Exiting." << std::endl;
				exit(1);
			}
			memcpy(coords, strand_start_ends_pos, (strand_start_ends_loc)*sizeof(hsize_t));

			if(H5Sselect_elements(channel_dataspace_id, H5S_SELECT_SET, strand_start_ends_loc, coords) < 0){  // Select the set of positions found from above
				std::cerr << "Unable to select set of elements from /StateData/" << channel_name << "/States dataset 'acquisition_raw_index' members from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
				H5Dclose(channel_dataset_id);
				H5Gclose(state_data_id_group);
				H5Fclose(file_id);
				free(channel_name);
				channel_name = NULL;
				free(channel_states);
				channel_states = NULL;
				free(strand_start_ends_pos);
				strand_start_ends_pos = NULL;
				free(coords);
				coords = NULL;
				return 0;
			}

			hsize_t total_strand_start_ends_loc = 0;
			if(strand_at_end){
				total_strand_start_ends_loc = strand_start_ends_loc+1;
			} else{
				total_strand_start_ends_loc = strand_start_ends_loc;
			}
			unsigned long long *strand_start_ends = (unsigned long long*) std::malloc(sizeof(unsigned long long)*total_strand_start_ends_loc);
			hsize_t *num_elements_needed = (hsize_t*) std::malloc(sizeof(hsize_t)*1);
			num_elements_needed[0] = strand_start_ends_loc;
			hid_t memSpace = H5Screate_simple(1, num_elements_needed, NULL);
			if(H5Dread(channel_dataset_id, dataspace_index_wanted, memSpace, channel_dataspace_id, H5P_DEFAULT, strand_start_ends) < 0){  // Read all start and end positions into an array
				std::cerr << "Could not get /StateData/" << channel_name << "/States dataset 'acquisition_raw_index' members from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
				H5Dclose(channel_dataset_id);
				H5Gclose(state_data_id_group);
				H5Fclose(file_id);
				free(channel_name);
				channel_name = NULL;
				free(channel_states);
				channel_states = NULL;
				free(strand_start_ends_pos);
				strand_start_ends_pos = NULL;
				free(strand_start_ends);
				strand_start_ends = NULL;
				free(coords);
				coords = NULL;
				free(num_elements_needed);
				num_elements_needed = NULL;
				return 0;
			}

			long long strand_start;
			long long strand_end;
			int num_positions_read = 0;
			if(strand_at_end){
				hid_t signal_dataset_id = H5Dopen(file_id, (CONCAT3("/Raw/",channel_name,"/Signal")).c_str(), H5P_DEFAULT); // Getting size of raw events read in and setting it to the last position
				if(signal_dataset_id < 0){
					std::cerr << "Could not get /Raw/" << channel_name << "/Signal for last element from bulk FAST5 (HDF5) file " << bulk5_file_name << ", aborting." << std::endl;
					H5Fclose(file_id);
					return 0;
				}
				hid_t signal_dataspace_id = H5Dget_space(signal_dataset_id);
				hid_t num_events_in_raw = H5Sget_simple_extent_npoints(signal_dataspace_id);
				strand_start_ends[total_strand_start_ends_loc-1] = num_events_in_raw; // Put size in last position
				H5Dclose(signal_dataset_id);
			}
			for(int i = 0; i < total_strand_start_ends_loc; i++){
				num_positions_read++;
				if(num_positions_read % 2 == 0){  // Every second value should be an end position
					strand_end = strand_start_ends[i];
					// std::cerr << "Start: " << strand_start << ", End: " << strand_end << std::endl;
					if(strand_start == strand_end){ //continue;
						if(i == total_strand_start_ends_loc-1){
							break;	// No further end to grab 
						}
						// std::cerr << "Start and end locations are the same! Finding the next possible end position:" << std::endl;
						unsigned long next_end_pos = coords[i] + 1;
						while(true){
							long long tmp_strand_end = getCoordVal(channel_dataset_id, dataspace_index_wanted, channel_dataspace_id, next_end_pos);
							if(tmp_strand_end > strand_start){
								strand_end = tmp_strand_end;
								break;
							}
							next_end_pos++;
						}
						// std::cerr << "Next end position is: " << strand_end << std::endl;
					}
					std::pair <char*, std::pair<long long,long long> > channel_start_end = std::make_pair(channel_name, std::make_pair(strand_start, strand_end));
					channel_ranges.push_back(channel_start_end);
					
					(*total_seqsize)++;
					strand_start = 0;
					strand_end = 0;
				} else{
					strand_start = strand_start_ends[i];
				}
			}
			free(coords);
			coords = NULL;
			free(strand_start_ends);
			strand_start_ends = NULL;
			free(num_elements_needed);
			num_elements_needed = NULL;
		}

		free(strand_start_ends_pos);
		strand_start_ends_pos = NULL;
		free(channel_states);
		channel_states = NULL;
		H5Dclose(channel_dataset_id);
	}
	H5Gclose(state_data_id_group);
	H5Fclose(file_id);
	
	return 1;
}
#endif

// Function that scans a fasta file and gets the number of sequences stored within it
// fasta_file_name - the fasta file we will be scanning
// num_sequences - a pointer that will store the total number of sequences found in the fasta file
// returns 1 on success. 0 on fail
// TODO: add error checking here in case the file doesn't exist. If this function does fail it needs to return a 0
inline int scan_fasta_data(const char *fasta_file_name, size_t *num_sequences){
	int local_seq_count_so_far = 0;
	// One sequence per line, values tab separated.
	std::ifstream f(fasta_file_name);
	
	if(!f.is_open()){
		std::cerr << "Error: FastA file could not be read. Please provide a valid FastA file." << std::endl;
		return 0;
	}
	
	std::string line;
    while (!f.eof()) {
      getline(f,line);
      if (line[0] == '>') // blank or header line
        local_seq_count_so_far++;
    }
    f.close();
	*num_sequences = local_seq_count_so_far;
	return 1;
}

// Function that scans a tsv file and gets the number of sequences stored within it
// text_file_name - the tsv file we will be scanning
// num_sequences - a pointer that will store the total number of sequences found in the fasta file
// returns 1 on success. 0 on fail
inline int scan_tsv_data(const char *text_file_name, size_t *num_sequences){

	// Count the number of lines in the file (buffering 1MB on read for speed) so we know how much space to allocate for sequence pointers 
	std::ios::sync_with_stdio(false); // optimization
	const int SZ = 1024 * 1024;
	std::vector <char> read_buffer( SZ );
	std::ifstream ifs(text_file_name, std::ios::binary); // Don't bother translating EOL as we are counting only, so using binary mode (PC + *NIX) 
	if(!ifs){
		return 0;
	}
	int n = 0;
	while(int sz = FileRead(ifs, read_buffer)) {
		n += CountLines(read_buffer, sz);
	}
	*num_sequences = n;
	return 1;
}

#endif
