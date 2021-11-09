# minioncapstone
Code related to functionality built in support of the McNUGGIT (MinION Capstone) Project 2017-19

## Overview

This project code consists of six major parts:

1. Code to calculate by GPU the Dynamic Time Warp (smallest distance between two discretely samples series of continuous values) of reference DNA and MinION output, both measured in picoAmps.
This calculation is approximate but suitable for the high entropy signal by partitioning the data stream into blocks (e.g. 16 picoAmp samples), Z-normalizing the query, and
calculating the random likelihood that co-linearity exists in the best matches of many co-linear blocks.  This is our golden standard of matching to which #2 should aspire in order to quickly
(<0.01s) match for filtering purposes (sending yea/nea to continue sequencing in a pore via the ONT Read-Until API).
2. Code to index reference sequences and match real data to them (CPU only) using matching of Discrete Cosine Transform (DCT) derived hash codes in co-linear blocks between query and reference
3. Empirically derived hash code matching frequency matrices to allow inexact matching of query and reference DCTs, and the scripts used to derive those matrices
4. A queue/prioritization system for data coming from the Read-Until API so that data taking too long to process is dropped (the older the data, the less valuable for accept/reject decisions).
5. A GUI (Web-based) to set the parameters for a MinION run's filtering (reference sequence, acceptable False Positive rate for negative selection or acceptable False Negative rate for positive filtering, % of sequences to apply the filter to)
6. A client that connects to the MinKNOW server. This client will receive reads from the MinKNOW device and the pass them to the DTW algorithm.

## Repository Status
The status of the following parts of the projects (as listed above) are as follows:

|      Contents         |    Status (Windows)   |     Status (Linux)    |
| :-------------------: | :-------------------: | :-------------------: |
|    DTW calculations   | Complete/ In Progress | Complete/ In Progress |
|        Indexing       |       In Progress     |       In Progress     |
|   Hash Code Matching  |       In Progress     |       In Progress     |
|    Read-Until Queue   |       Incomplete      |       Incomplete      |
|           GUI         |       Incomplete      |       Incomplete      |
|   Read-Until Client   |       In Progress     |       In Progress     |

## Dependencies (Linux)
This repository requires that you have the following installed to compile properly:
- [Boost C++ Libraries](https://www.boost.org/)
- [CUDA Development Tools](https://developer.nvidia.com/cuda-toolkit) are required to use nvcc
- All Linux dependencies for building gRPC as listed [here](grpc/INSTALL.md)

## Dependencies (Windows)
This repository requires that you have the following installed to compile properly:
- [Boost C++ Libraries](https://www.boost.org/)
    - Windows currently uses version 1.70.0 of boost
    - Ensure that boost is extracted to "C:\Program Files\boost" directory
- [HDF5 development libraries](https://www.hdfgroup.org/downloads/hdf5/)
- [Visual Studio 2015](https://visualstudio.microsoft.com/vs/older-downloads/)
- [Dirent API for Microsoft Visual Studio](https://web.archive.org/web/20170428133315/http://www.softagalleria.net/dirent.php). This must be put in the "include" directory Visual Studio uses (ex: C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include)
- [CUDA Development Tools](https://developer.nvidia.com/cuda-toolkit) are required to use nvcc
- Make for Windows (recommended for building executables). Some programs that work include:
     - [Cygwin](https://www.cygwin.com/)
     - [MinGW](http://www.mingw.org/)
- All Windows dependencies for building gRPC with CMake as listed [here](grpc/INSTALL.md)

Ensure that your path has been updated appropriately following the installation of all above dependencies

## Building (Linux)
To build the grpc libraries, run the following script:
```
$ ./all_build.sh
```
**Note**: all_build.sh will automatically build the grpc libraries in minioncapstone/bin. If you want to specify a different path, you can pass that as an argument like so:
```
$ ./all_build.sh /full/path/to/library/install/
```
Be sure to add the lib directory found in bin to your LD_LIBRARY_PATH after running:
```
/Directory/path/to/minioncapstone/bin/lib/
```

Make the various search, index and client executables:
```
$ make all_linux
```

## Building (Windows)
Build grpc libraries:
```
C:\Directory\path\to\minioncapstone> build_all.bat
```
Make the search, index and client executables:
```
C:\Directory\path\to\minioncapstone> make all_windows
```
Add the lib directory to your PATH:
```
C:\Directory\path\to\minioncapstone\lib
```

## ont_simple_client
Receives and prints reads received from the MinKNOW server. Performs matching against a reference genome (reference genome can be a pre indexed file or a FastA file that will be indexed upon running ont_simple_client. Additionally, the FastA file can be prefixed). CURRENTLY MATCHING DOES NOT WORK. The options for running ont_simple_client are as follows:

Options for reading BED file:
- -B (Read in BED file to compare matches to. Must provide file path with this option)

Options relating to connecting and reading in values are:
- -H (Host to open a connection on. Default is "localhost")
- -P (Port to connect to] default. Default is 8004)
- -S (Start channel to get reads from. Default is 1)
- -E (End channel to get reads from. Default is 512)
- -g (Size of the buffers that will be used to store reads for each pore. Default is 24000)
- -u (Upper limit to check for which determines if a read is instrand. Default is 1300)
- -l (Lower limit to check for which determines if a read is instrand. Default is 300)
- -t (Number of threads reads will be processed on. Default will be set to the number of GPUs on your device)

Options related to matches are:
- -b (Input is binary)
- -p (P-value limit for reporting matches i.e. anchor DTW distance Mann-Whitney test random match probability for the subject DB used, floating point. Default is 0.01)
- -f (FDR limit for reporting matches i.e. Benjamini-Hochberg multiple-testing corrected p-value for the subject DB used, floating point. Default is 1)
- -r (Ranks limit for reporting matches i.e. the number of matches to report that pass the pvalue and FDR criteria. Guaranteed to be the best matches unless # passing criteria are > 2^20. Default is 100)
- -w (Warp max. The proportion of length deviation allowed between query and subject in alignment, larger=more sensitive & longer runtime, floating point. Default is 0.25)
- -s (Segment size mean, which helps define average event size for segmentation, floating point. Default is 66.6)
- -a (Attenuation limit for a segment i.e. segments longer than this multiple of the mean segment size get split, floating point. Default is 3.0)

Options related to FastA indexing are:
- -i (Index FastA file. Must provide file path with this option. The "reference_genome" argument would be replaced by the file path here)
- -T (single strand indexing only)
- -n (Use standard deviation for signal type in FastA files. Default uses mean)
- -c (Also generate signal for the reverse complement strand)
- -C (Exclude default forward strand encoding)
- -R (Convert input as RNA. Default is DNA)

Options related to FastA prefixing are:
- -y (Prefix FastA sequences)
- -z (Reverse the sequence. E.g. for nanopore 3'->5' direct RNA analysis)
- -s (Unique suffix length to include in the output. Default is 100)
- -m (Minimal output)

Additional options are:
- -v (Verbose mode)
- -h (Help message which displays all information above)

### Running ont_simple_client (Linux)
#### Step 1. Run MinKNOW and start up a run
- The run can either be a real run or one sent through the playback script

#### 2. Run ont_simple_client.exe

Example:
```
$ ./ont_simple_client -s 20 -e 120 -v
```

Note: Reads will be obtained from the start channel to the end channel. End channel must not be smaller than start channel

#### 2a. Run ont_simple_client.exe when obtaining reads from a server
ont_simple_client can also connect to a specified host and port if you wish to obtain reads from a MinKNOW that's running elsewhere (not connected to your current machine). This is done with the -H and -P flags (for host and port, respectively)

Example:
```
$ ./ont_simple_client -H 987.654.321.000 -P 4321
```

### Running ont_simple_client.exe (Windows)
#### Step 1. Run MinKNOW and start up a run
- The run can either be a real run or one sent through the playback script
- Skip to step 2 if playback script is not needed

#### Step 1a. Run the playback script (if needed)
Example:
```
C:\Directory\path> "C:\Program Files\OxfordNanopore\MinKNOW\ont-python\python.exe" "C:\Program Files\OxfordNanopore\MinKNOW\python\recipes\playback.py" --source <bulk/fast5/file/path.fast5>
```
Note: Playback script **must** be run with Oxford Nanopores own python executable

#### 2. Run ont_simple_client.exe

Example:
```
C:\Directory\path\to\minioncapstone> ont_simple_client.exe -s 100 -e 200 -v
```

Note: Reads will be obtained from the start channel to the end channel. End channel must not be smaller than start channel

#### 2a. Run ont_simple_client.exe when obtaining reads from a server
ont_simple_client can also connect to a specified host and port if you wish to obtain reads from a MinKNOW that's running elsewhere (not connected to your current machine). This is done with the -H and -P flags (for host and port, respectively)

Example:
```
C:\Directory\path\to\minioncapstone> ont_simple_client.exe -H 123.456.789.000 -P 1234
```

## check_bulk_data
Reads a Bulk Fast5 file and prints the data specified.

check_bulk_data takes the following arguments:
- Bulk Fast5 file path
- Channel number to check data from (must be between 1 and 512)
- Position in the data to start checking reads from
- The number of reads to check

### Running check_bulk_data (Linux)
Example:
```
$ ./check_bulk_data /Directory/path/to/bulk.fast5 401 601090 1500
```
The above example will return 1500 reads starting at position 601090 from channel 401

### Running check_bulk_data.exe (Windows)
Example:
```
C:\Directory\path\to\minioncapstone> check_bulk_data.exe C:\Directory\path\to\bulk.fast5 100 512000 1200
```
The above example will return 1200 reads starting at position 512000 from channel 100

## magenta_short_index
Indexes either a FastA, directory of Fast5 files, or a Bulk Fast5 file and writes the output to a set of files. These files are reference genomes that will be used for the matching in magenta and ont_simple_client. Data in files generated can either be in floats or shorts. Options for magenta_short_index are as follows:

Options for selecting index input file type are (must run with **one** of the following):
- -a (Index a bulk FastA file)
- -l (Index a bulk Fast5 file)
- -f (Index a set of Fast5 files in a directory)

Options for indexing are:
- -u (Use raw data when reading in data. Raw data is read in as shorts. Will read in as floats if not used)
- -g (Segment data read in. Currently not used)

Options related to FastA indexing are:
- -r (Convert input as RNA when indexing FastA files. Default is DNA)
- -c (Also generate signal for the reverse complement strand)
- -C (Exclude default forward strand encoding)
- -n (Use standard deviation for signal type in FastA files. Default uses mean)
- -b (Block size for indexing. Currently not used. Default is 17)

Additional options are:
- -v (Verbose mode)
- -h (Help message which displays all information above)

### Running magenta_short_index (Linux)
An output path filename is taken as an argument. Given an output path of /Directory/path/to/output the following files will be created:
- /Directory/path/to/output.txt - Contains the indexed values
- /Directory/path/to/output.hpr - Contains the raw half point values
- /Directory/path/to/output-sub.txt - Contains the non-indexed subject in a txt file

Example 1 - Indexing a FastA file:
```
$ ./magenta_short_index -a -v /Directory/path/to/reference_nucleotides.fasta /Directory/path/to/output
```

The above example will index the FastA file in verbose mode.

Example 2 - Indexing a Bulk Fast5 file:
```
$ ./magenta_short_index -l -u /Directory/path/to/bulk_file.fast5 /Directory/path/to/output
```

The above example will index the Bulk Fast5 file using raw data that will be written as shorts

Example 3 - Indexing a Directory of Fast5 Files:
```
$ ./magenta_short_index -f /Directory/path/to/Fast5/files/ /Directory/path/to/output
```
The above example will index the directory of Fast5 files using non-raw data since the -u flag was not provided. Data will be written as floats

### Running magenta_short_index.exe (Windows)
An output path filename is taken as an argument. Given an output path of C:\Directory\path\to\output the following files will be created:
- C:\Directory\path\to\output.txt - Contains the indexed values
- C:\Directory\path\to\output.hpr - Contains the raw half point values
- C:\Directory\path\to\output-sub.txt - Contains the non-indexed subject in a txt file


Example 1 - Indexing a FastA file:
```
C:\Directory\path\to\minioncapstone> magenta_short_index.exe -a -v -s C:\Directory\path\to\reference_nucleotides.fasta C:\Directory\path\to\output
```

The above example will index the FastA file in verbose mode single stranded.

Example 2 - Indexing a Bulk Fast5 file:
```
C:\Directory\path\to\minioncapstone> magenta_short_index.exe -l -u C:\Directory\path\to\bulk_file.fast5 C:\Directory\path\to\output
```

The above example will index the Bulk Fast5 file using raw data that will be written as shorts

Example 3 - Indexing a Directory of Fast5 Files:
```
C:\Directory\path\to\minioncapstone> magenta_short_index.exe -f C:\Directory\path\to\Fast5\files\ C:\Directory\path\to\output
```
The above example will index the directory of Fast5 files using non-raw data since the -u flag was not provided. Data will be written as floats

## magenta
Receives a FastA, directory of Fast5, or Bulk5 file and compares it to either a reference file generated by magenta_short_index or a FastA, directory of Fast5, or Bulk5 file. Any match that is found (based on the false discovery rate that can be set by the user) is written to an output file. **CURRENTLY DOES NOT WORK ON EITHER OS**

magenta is run with the following arguments:
- The indexed reference file built by magenta_short_index (Without the extensions as magenta uses the different output files that magenta_short_index made), OR
- A FastA, directory of Fast5, or Bulk5 file as a reference
- The path that magenta will write output containing the matches to
- A FastA, directory of Fast5, or Bulk5 file to compare to the reference file provided above

Options for running magenta are as follows:

Options for reading BED file:
- -B (Read in BED file to compare matches to. Must provide file path with this option)

Options related to matches are:
- -b (Input is binary)
- -u (Use raw data. Raw data is output as shorts. Default uses event data, which are floats)
- -p (P-value limit for reporting matches i.e. anchor DTW distance Mann-Whitney test random match probability for the subject DB used, floating point. Default is 0.01)
- -f (FDR limit for reporting matches i.e. Benjamini-Hochberg multiple-testing corrected p-value for the subject DB used, floating point. Default is 1)
- -r (Ranks limit for reporting matches i.e. the number of matches to report that pass the pvalue and FDR criteria. Guaranteed to be the best matches unless # passing criteria are > 2^20. Default is 100)
- -w (Warp max. The proportion of length deviation allowed between query and subject in alignment, larger=more sensitive & longer runtime, floating point. Default is 0.25)
- -s (Segment size mean, which helps define average event size for segmentation, floating point. Default is 66.6)
- -a (Attenuation limit for a segment i.e. segments longer than this multiple of the mean segment size get split, floating point. Default is 3.0)

Options related to using a FastA file:
- -i (Use FastA file as reference. Must provide file path with this option)
- -S (Single strand indexing only)
- -n (Use standard deviation for signal type in FastA files. Default uses mean)
- -c (Also generate signal for the reverse complement strand)
- -C (Exclude default forward strand encoding)
- -R (Convert input as RNA when indexing FastA files. Default is DNA)
- -I (Use FastA file as a query)

Options related to using Fast5 files:
- -e (Use directory of Fast5 files as reference. Must provide file path with this option)
- -E (Use directory of Fast5 files as a query)

Options related to using a Bulk5 file:
- -l (Use Bulk5 file as reference. Must provide file path with this option)
- -L (Use Bulk5 file as a query)

Additional options are:
- -v (Verbose mode)
- -h (Help message which displays all information above)

### Running magenta (Linux)
Coming soon (In development)

Example:
```
$ ./magenta -v -B /Directory/path/to/file.bed -E -i /Directory/path/to/file.fasta /Directory/path/to/output /Directory/path/to/fast5/files/
```

The above example will compare the query of Fast5 files against the file.fasta reference and check for any overlaps with intervals read in from the BED file file.bed

### Running magenta.exe (Windows)

Example:
```
C:\Directory\path\to\minioncapstone> magenta.exe -p 0.05 -w 0.5 -I C:\Directory\path\to\indexed_reference C:\Directory\path\to\output C:\Directory\path\to\reference_nucleotides.fasta
```
The above example will compare the query reference_nucleotides.fasta file against the indexed_reference with a false discovery rate of 0.05 and a warp of 0.5

## Testing
Tests for all functions used can be found in the tests/ directory. To build and run the test executable, do the following:

### magenta_utils_tests (Linux)

Move into the tests/ directory

```
$ cd tests/
```

Make magenta_utils_tests

```
$ make magenta_utils_tests
```

Run magenta_utils_tests

```
$ ./magenta_utils_tests
```

Confirm that all tests run properly with no errors.

### magenta_utils_tests.exe (Windows)

Move into the tests\ directory

```
C:\Directory\path\to\minioncapstone> cd tests\
```

Make magenta_utils_tests

```
C:\Directory\path\to\minioncapstone\tests> make magenta_utils_tests.exe
```

Run magenta_utils_tests

```
C:\Directory\path\to\minioncapstone\tests> magenta_utils_tests.exe
```

Confirm that all tests run properly with no errors.

## Related
Slack: https://minioncapstone.slack.com (commits are notified here too, you can use screen share, whiteboard, other cool stuff)

Confluence: https://minioncapstone.atlassian.net (collaborative planning and documentation)

gRPC: https://github.com/grpc/grpc

pthreads for Windows build: https://github.com/GerHobbelt/pthread-win32

## Credits
- all_build.bat and edit_props.ps1 based off of [plasticbox\grpc-windows](https://github.com/plasticbox/grpc-windows)
- proto files obtained from [minkow-api]
- threads.cpp, threads.h and wqueue.h obtained and modified from [vichargrave/wqueue](https://github.com/vichargrave/wqueue)
- IntervalTree.h obtained from [ekg/intervaltree
](https://github.com/ekg/intervaltree)
- Catch2 single header version obtained from [catchorg/Catch2](https://github.com/catchorg/Catch2)
- triemap obtained and modified from [RedisLabs/triemap](https://github.com/RedisLabs/triemap)