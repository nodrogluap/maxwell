struct refGenome {
    float* Sub;
    float* Avg;
    float* Std;
    int M;
    int N;
    int single_strand;
    int ignore_from;
    int ignore_to;
};

// Need the following for building DLLs in Windows
#if defined(_WIN32) && defined(_COMPILING_CDTW_DLL)
	#ifndef MYDLL_H
	#define MYDLL_H

	// extern "C" __declspec(dllexport)
	#define EXPORT_CDTW __declspec(dllexport)

	// Must move all functions needed in DLL in here
	EXPORT_CDTW std::pair<float, size_t> prune_cdtw_clean(float *, float *, float *, int, int, int);
	EXPORT_CDTW struct refGenome load_ref_genome_clean(char *, int, int, int);
	EXPORT_CDTW int initialize_device_clean(int device_num);
	EXPORT_CDTW int get_num_devices_clean();
	EXPORT_CDTW int freeBuffers();
	#endif
// No need for DLLs if not on Windows
#else
	// Empty defines
	#define EXPORT_CDTW
	std::pair<float, size_t> prune_cdtw_clean(float *, float *, float *, int, int, int);
	struct refGenome load_ref_genome_clean(char *, int, int, int);
	int initialize_device_clean(int device_num);
	int get_num_devices_clean();
	int freeBuffers();
#endif

struct refGenome load_ref_genome_tmp();
// void cuda_dealloc();
