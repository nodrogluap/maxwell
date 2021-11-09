#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <string>
#include <algorithm>

#define CONCAT3(a,b,c)   std::string(a)+std::string(b)+std::string(c)
// using namespace std;
extern "C"{
#include "hdf5.h"
}

unsigned int FileRead( std::istream & is, std::vector <char> & buff ) {
    is.read( &buff[0], buff.size() );
    return is.gcount();
}

unsigned int CountLines( const std::vector <char> & buff, int sz ) {
    int newlines = 0;
    const char * p = &buff[0];
    for ( int i = 0; i < sz; i++ ) {
        if ( p[i] == '\n' ) {
            newlines++;
        }
    }
    return newlines;
}

template <class T>
T* shortToTemplate(short* data, long long data_length){
	T* return_data;
	return_data = (T*)malloc(sizeof(T)*data_length);
	std::transform(data, data + data_length, return_data, [](short s){ return (T)s; });
	return return_data;
}

template <class T>
int
read_binary_data(const char *binary_file_name, T **output_vals, unsigned long long int *num_output_vals){

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
    return 1;
  }

  T *out = (T *) malloc(sizeof(T)*n);
  if(out == 0){
    std::cerr << "Cannot allocate CPU memory for subject (" << sizeof(T) << "*" << n << " bytes)" << std::endl;
    return 1;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read((char *) out, n);

  // Only set the output if all the data was succesfully read in.
  *output_vals = out;
  return 0;
}

template <class T>
int
read_text_data(const char *text_file_name, T **output_vals, unsigned long long int *num_output_vals){
 
  // Count the number of lines in the file (buffering 1MB on read for speed) so we now how much space to allocate for output_vals
  // std::ios::sync_with_stdio(false); //optimization
  const int SZ = 1024 * 1024;
  std::vector <char> read_buffer( SZ );
  std::ifstream ifs(text_file_name, std::ios::binary); // Don't bother translating EOL as we are counting only, so using binary mode (PC + *NIX) 
  if(!ifs){
    std::cerr << "Error reading in file " << text_file_name << " exiting" << std::endl;
    return 1;
  }
  int n = 0;
  while(int sz = FileRead(ifs, read_buffer)) {
    n += CountLines(read_buffer, sz);
  }
  *num_output_vals = n;
  if(n == 0){
    std::cerr << "File is empty or not proparly formatted. Exiting." << std::endl;
  ifs.close();
    return 1;
  }

  T *out = (T *) malloc(sizeof(T)*n);
  if(out == 0){
    std::cerr << "Cannot allocate CPU memory for subject (" << sizeof(T) << "*" << n << " bytes)" << std::endl;
  ifs.close();
    return 1;
  }
  
  // Read the actual values
  ifs.clear(); // get rid of EOF error state
  ifs.seekg(0, std::ios::beg);
  std::stringstream in;      // Make a stream for the line itself
  std::string line;
  int i = 0;
  while(n--){  // Read line by line
    std::getline(ifs, line); in.str(line);
    in >> out[i++];      // Read the first whitespace-separated token
    in.clear(); // to reuse the stringatream parser
  }

  // Only set the output if all the data was succesfully read in.
  *output_vals = out;
  ifs.close();
  return 0;
}

template <class T>
int
read_fast5_data(const char *fast5_file_name, T **output_vals, unsigned long long int *num_output_vals){
 
  // char* sample_name;
  // long long buffer_size;
  // double sample_rate;
  // T* out = readEventsFromFast5<T>(fast5_file_name, &buffer_size, &sample_name, &sample_rate, use_raw, 0);
  // *output_vals = out;
  // return 0;

  /* Open an existing file. */
  hid_t file_id = H5Fopen(fast5_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0){
    std::cerr << "Could not open HDF5 file " << fast5_file_name << " so skipping" << std::endl;
    return 0;
  }

  // Getting groups
  hid_t channel_id_group = H5Gopen(file_id, "/UniqueGlobalKey/channel_id", H5P_DEFAULT);
  if(channel_id_group < 0){
    std::cerr << "Could not get /UniqueGlobalKey/channel_id group from FAST5 (HDF5) file " << fast5_file_name << std::endl;
    H5Fclose(file_id);
    return 0;
  }

  hid_t read_group = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);

  if(read_group < 0){
    std::cerr << "Could not get /Raw/Reads group from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
    H5Gclose(channel_id_group);
    H5Fclose(file_id);
    return 0;
  }

  hid_t read_group_for_name = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
  if(read_group_for_name < 0){
    std::cerr << "Could not get any Reads group from FAST5 (HDF5) file for name" << std::endl;
    H5Gclose(read_group);
    H5Gclose(channel_id_group);
    H5Fclose(file_id);
    return 0;
  }

  hsize_t num_read_objects;
  if(H5Gget_num_objs(read_group, &num_read_objects)){
    std::cerr << "Could not get /Raw/Reads group objects from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
    H5Gclose(read_group_for_name);
    H5Gclose(read_group);
    H5Gclose(channel_id_group);
    H5Fclose(file_id);
    return 0;
  }

  std::cerr << "num read objects: " << num_read_objects << std::endl;
  // See if the object looks like a read based on its name
  ssize_t name_size = H5Gget_objname_by_idx(read_group_for_name, 0, NULL, 0);
  name_size++; // for NULL termination
  char *read_subgroup_name = NULL;
  errno = 0;
  read_subgroup_name = (char *) std::malloc(name_size);
  if(errno || (read_subgroup_name == NULL)){
    std::cerr << "Error in malloc for HDF5 group name: " << strerror(errno) << std::endl;
    exit(1);
  }
  H5Gget_objname_by_idx(read_group_for_name, 0, read_subgroup_name, (size_t) name_size);
  if(name_size < 5 || read_subgroup_name[0] != 'R' || read_subgroup_name[1] != 'e' || read_subgroup_name[2] != 'a' || read_subgroup_name[3] != 'd' || read_subgroup_name[4] != '_'){ // should have the form Read_#
    std::cout << "Skipping unexpected HDF5 object " << read_subgroup_name << std::endl;
    free(read_subgroup_name);
    read_subgroup_name = NULL;
    exit(1);
  }

  /* Open an existing read's dataset. */
  hid_t samples_dataset_id = H5Dopen(file_id, (CONCAT3("/Raw/Reads/",read_subgroup_name,"/Signal")).c_str(), H5P_DEFAULT);
  hid_t samples_dataspace_id = H5Dget_space(samples_dataset_id);    /* dataspace handle */
  hsize_t num_samples = (int)H5Sget_simple_extent_npoints(samples_dataspace_id); //NOTE: assumes below that it's one dimensional

  if(num_samples < 1){
    std::cerr << "Could not get dimensions of /Raw/Reads/" << read_subgroup_name << "/Signal dataspace from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
    free(read_subgroup_name);
    read_subgroup_name = NULL;
    exit(1);;
  }

  errno = 0;
  short* out = (short *) std::malloc(sizeof(short)*num_samples);
  if(errno || (out == NULL)){
    std::cerr << "Error in malloc for HDF5 sample data (" << num_samples << " samples): " << strerror(errno) << std::endl;
    exit(1);
  }
  if(H5Dread(samples_dataset_id, H5T_STD_I16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0){
    std::cerr << "Could not get /Raw/Reads/" << read_subgroup_name << "/Signal dataset members from FAST5 (HDF5) file " << fast5_file_name << " so skipping" << std::endl;
    exit(1);;
  }

  H5Dclose(samples_dataset_id);

  *num_output_vals = num_samples;
  H5Gclose(read_group_for_name);
  H5Gclose(read_group);
  H5Gclose(channel_id_group);
  H5Fclose(file_id);
  free(read_subgroup_name);
  read_subgroup_name = NULL;

  *output_vals = shortToTemplate<T>(out, num_samples);
  return 0;
}

template <class T>
int
read_data(const char *file_name, T **output_vals, unsigned long long int *num_output_vals){
  char filename[256];
  strcpy(filename, file_name);
  char *extension;
  char *ptr = strtok(filename, ".");
  while(ptr = strtok(NULL, ".")) {
    extension = ptr;
  }
  std::cerr << "extension: " << extension << std::endl;
  if(strcmp(extension, "txt") == 0) return read_text_data(file_name, output_vals, num_output_vals);
  else if(strcmp(extension, "fast5") == 0) return read_fast5_data(file_name, output_vals, num_output_vals);
  return read_binary_data(file_name, output_vals, num_output_vals);
}
