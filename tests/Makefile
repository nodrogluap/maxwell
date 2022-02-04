#----General Definitions----#

MAX_DIR := ${CURDIR}/..

#----Compilers----#
NVCC=nvcc

#----Debug----#
DEBUG=-g
CUDA_DEBUG=-G
FLASH_TEST_DEBUG=-D_DEBUG_REG

#----Linux----#

#----Include Directories----#

INCLUDE_HDF5_LINUX=$(MAX_DIR)/submodules/hdf5/hdf5/include

HDF5_LIB_LINUX=$(MAX_DIR)/submodules/hdf5/hdf5/lib
ZLIB_LIB_LINUX=$(MAX_DIR)/submodules/grpc/cmake/build/third_party/zlib

VBZ_LIB_LINUX=$(MAX_DIR)/submodules/vbz_compression/build/lib

ZLIB_FLAGS_LINUX=-l:libz.a
HDF5_LIBNAME=libhdf5.a

HDF5_FLAGS_LINUX=-l:$(HDF5_LIBNAME)

VBZ_LIBNAME=libvbz.a
VBZ_FLAGS_LINUX=-l:$(VBZ_LIBNAME)

#----Linux Flags----#

CXX11=-std=c++11

#----make objects for linux----#

flash_dtw_test: flash_dtw_test.cu ../flash_dtw.cuh ../cuda_utils.hpp ../cuda_utils.h flash_dtw_test_utils.cuh
	$(NVCC) $(CXX11) $(FLASH_TEST_DEBUG) $(DEBUG) $(CUDA_DEBUG) -I$(INCLUDE_HDF5_LINUX) flash_dtw_test.cu -o flash_dtw_test
	
segmentation_test: segmentation_test.cu ../segmentation.hpp ../all_utils.hpp
	$(NVCC) $(CXX11) $(DEBUG) $(CUDA_DEBUG) segmentation_test.cu -o segmentation_test
	
all_utils_test: all_utils_test.cu ../all_utils.hpp
	$(NVCC) $(CXX11) $(DEBUG) $(CUDA_DEBUG) -I$(INCLUDE_HDF5_LINUX) -DHDF5_SUPPORTED=1 all_utils_test.cu -o all_utils_test -L$(ZLIB_LIB_LINUX) $(ZLIB_FLAGS_LINUX) -L$(HDF5_LIB_LINUX) $(HDF5_FLAGS_LINUX) -L$(VBZ_LIB_LINUX) $(VBZ_FLAGS_LINUX)
	
cuda_utils_test: cuda_utils_test.cu ../cuda_utils.hpp ../all_utils.hpp ../ont_utils.cuh
	$(NVCC) $(CXX11) $(DEBUG) $(CUDA_DEBUG) -I$(INCLUDE_HDF5_LINUX) -DHDF5_SUPPORTED=1 cuda_utils_test.cu -o cuda_utils_test -L$(ZLIB_LIB_LINUX) $(ZLIB_FLAGS_LINUX) -L$(HDF5_LIB_LINUX) $(HDF5_FLAGS_LINUX) -L$(VBZ_LIB_LINUX) $(VBZ_FLAGS_LINUX)
	
#----Windows----#

#----Include Directories----#

INCLUDE_HOME=../.
INCLUDE=../include

#----Include Flags----#

#----make objects for windows----#

flash_utils_test.exe: flash_utils_test.cu
	$(NVCC) $(DEBUG) -I$(INCLUDE_HOME) -I$(INCLUDE) flash_utils_test.cu -o flash_utils_test.exe

flash_dtw_test.exe: flash_dtw_test.cu ../flash_dtw.cuh ../flash_utils.hpp ../cuda_utils.h
	$(NVCC) $(DEBUG) -I$(INCLUDE_HOME) -I$(INCLUDE) flash_dtw_test.cu -o flash_dtw_test.exe
