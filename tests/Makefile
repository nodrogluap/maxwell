#----General Definitions----#

#----Compilers----#
NVCC=nvcc

#----Debug----#
DEBUG=-g
CUDA_DEBUG=-G
FLASH_TEST_DEBUG=-D_DEBUG_REG

#----Linux----#

#----Include Directories----#

#----Linux Flags----#

CXX11=-std=c++11

#----make objects for linux----#

flash_utils_test: flash_utils_test.cu ../flash_utils.hpp
	$(NVCC) $(CXX11) $(DEBUG) $(CUDA_DEBUG) flash_utils_test.cu -o flash_utils_test

flash_dtw_test: flash_dtw_test.cu ../flash_dtw.cuh ../flash_utils.hpp ../cuda_utils.h flash_dtw_test_utils.cuh
	$(NVCC) $(CXX11) $(FLASH_TEST_DEBUG) $(DEBUG) $(CUDA_DEBUG) flash_dtw_test.cu -o flash_dtw_test
	
segmentation_test: segmentation_test.cu ../segmentation.hpp ../all_utils.hpp
	$(NVCC) $(CXX11) $(DEBUG) $(CUDA_DEBUG) segmentation_test.cu -o segmentation_test
	
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
