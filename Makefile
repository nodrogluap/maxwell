#----General Definitions----#

PROGNAME=maxwell

#----Compilers----#

NVCC= nvcc
CC=gcc
# NVCC2=/usr/local/cuda-10.1/bin/nvcc

#----Compiler Flags----#

DEBUG=-g
CUDA_DEBUG=-G

#----Directories----#

DIR := ${CURDIR}
INCLUDE=$(DIR)/include
RPC=include/minknow_api

#----ont flags----#

# ONT_CLIENT_FILES=$(RPC)/acquisition.grpc.pb.cc $(RPC)/acquisition.pb.cc $(RPC)/analysis_configuration.grpc.pb.cc $(RPC)/analysis_configuration.pb.cc $(RPC)/data.grpc.pb.cc $(RPC)/data.pb.cc $(RPC)/device.grpc.pb.cc $(RPC)/device.pb.cc $(RPC)/instance.grpc.pb.cc $(RPC)/instance.pb.cc $(RPC)/keystore.grpc.pb.cc $(RPC)/keystore.pb.cc $(RPC)/log.grpc.pb.cc $(RPC)/log.pb.cc $(RPC)/manager.grpc.pb.cc $(RPC)/manager.pb.cc $(RPC)/minion_device.grpc.pb.cc $(RPC)/minion_device.pb.cc $(RPC)/promethion_device.grpc.pb.cc $(RPC)/promethion_device.pb.cc $(RPC)/protocol.grpc.pb.cc $(RPC)/protocol.pb.cc $(RPC)/rpc_options.pb.cc $(RPC)/statistics.grpc.pb.cc $(RPC)/statistics.pb.cc
ONT_CLIENT_FILES=$(RPC)/acquisition.grpc.pb.o $(RPC)/acquisition.pb.o $(RPC)/analysis_configuration.grpc.pb.o $(RPC)/analysis_configuration.pb.o $(RPC)/data.grpc.pb.o $(RPC)/data.pb.o $(RPC)/device.grpc.pb.o $(RPC)/device.pb.o $(RPC)/instance.grpc.pb.o $(RPC)/instance.pb.o $(RPC)/keystore.grpc.pb.o $(RPC)/keystore.pb.o $(RPC)/log.grpc.pb.o $(RPC)/log.pb.o $(RPC)/manager.grpc.pb.o $(RPC)/manager.pb.o $(RPC)/minion_device.grpc.pb.o $(RPC)/minion_device.pb.o $(RPC)/promethion_device.grpc.pb.o $(RPC)/promethion_device.pb.o $(RPC)/protocol.grpc.pb.o $(RPC)/protocol.pb.o $(RPC)/rpc_options.pb.o $(RPC)/statistics.grpc.pb.o $(RPC)/statistics.pb.o

#----Linux----#

#----Include Directories----#

SO_LIB_PATH=$(DIR)/bin/lib
HDF5_LIB_LINUX=$(DIR)/submodules/hdf5/hdf5/lib
ZLIB_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/zlib

INCLUDE_GRPC_LINUX=$(DIR)/submodules/grpc/include
INCLUDE_ABSEIL_LINUX=$(DIR)/submodules/grpc/third_party/abseil-cpp
INCLUDE_GOOGLE_LINUX=$(DIR)/submodules/grpc/third_party/protobuf/src
INCLUDE_HDF5_LINUX=$(DIR)/submodules/hdf5/hdf5/include

GRPC_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build
PROTOBUF_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/protobuf
ABSL_BASE_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/base
ABSL_CONTAINER_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/container
ABSL_DEBUG_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/debugging
ABSL_FLAGS_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/flags
ABSL_HASH_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/hash
ABSL_NUM_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/numeric
ABSL_RAN_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/random
ABSL_STAT_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/status
ABSL_STRINGS_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/strings
ABSL_SYNC_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/synchronization
ABSL_TIME_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/time
ABSL_TYPES_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/abseil-cpp/absl/types

RE2_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/re2

BORING_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/boringssl-with-bazel
CARES_LIB_LINUX=$(DIR)/submodules/grpc/cmake/build/third_party/cares/cares/lib

VBZ_LIB_LINUX=$(DIR)/submodules/vbz_compression/build/lib

#----Include Flags----#

HDF5_LIBNAME=libhdf5.a

HDF5_FLAGS_LINUX=-l:$(HDF5_LIBNAME)
ZLIB_FLAGS_LINUX=-l:libz.a
GRPC_FLAGS_LINUX=-l:libgrpc++.a -l:libgrpc.a -l:libaddress_sorting.a -l:libupb.a -l:libgpr.a
PROTO_FLAGS_LINUX=-l:libprotobuf.a -l:libprotoc.a
# ABSL_BASE_FLAGS_LINUX=-labsl_base -labsl_log_severity -labsl_periodic_sampler -labsl_scoped_set_env -labsl_strerror -labsl_exponential_biased -labsl_malloc_internal -labsl_raw_logging_internal -labsl_spinlock_wait -labsl_throw_delegate
ABSL_BASE_FLAGS_LINUX=-l:libabsl_spinlock_wait.a -l:libabsl_throw_delegate.a -l:libabsl_base.a -l:libabsl_malloc_internal.a -l:libabsl_raw_logging_internal.a
# ABSL_CONTAINER_FLAGS_LINUX=-labsl_hashtablez_sampler -labsl_raw_hash_set
ABSL_CONTAINER_FLAGS_LINUX=-l:libabsl_raw_hash_set.a
# ABSL_DEBUGGING_FLAGS_LINUX=-labsl_debugging_internal -labsl_examine_stack -labsl_leak_check -labsl_stacktrace -labsl_demangle_internal -labsl_failure_signal_handler -labsl_leak_check_disable -labsl_symbolize
 ABSL_DEBUG_FLAGS_LINUX= -l:libabsl_symbolize.a -l:libabsl_stacktrace.a -l:libabsl_demangle_internal.a -l:libabsl_debugging_internal.a
# ABSL_FLAGS_FLAGS_LINUX=-labsl_flags_commandlineflag -labsl_flags_internal -labsl_flags_private_handle_accessor -labsl_flags_usage -labsl_flags_commandlineflag_internal -labsl_flags_marshalling -labsl_flags_program_name -labsl_flags_usage_internal -labsl_flags -labsl_flags_config -labsl_flags_parse -labsl_flags_reflection
ABSL_FLAGS_FLAGS_LINUX=-l:libabsl_flags_commandlineflag.a -l:libabsl_flags_internal.a -l:libabsl_flags_private_handle_accessor.a -l:libabsl_flags_usage.a -l:libabsl_flags_commandlineflag_internal.a -l:libabsl_flags_marshalling.a -l:libabsl_flags_program_name.a -l:libabsl_flags_usage_internal.a -l:libabsl_flags.a -l:libabsl_flags_config.a -l:libabsl_flags_parse.a -l:libabsl_flags_reflection.a
# ABSL_HASH_FLAGS_LINUX=-labsl_city -labsl_hash -labsl_wyhash
ABSL_HASH_FLAGS_LINUX=-l:libabsl_hash.a -l:libabsl_city.a -l:libabsl_wyhash.a
ABSL_NUM_FLAGS_LINUX=-l:libabsl_int128.a
# ABSL_RAN_FLAGS_LINUX=-labsl_random_internal_distribution_test_util -labsl_random_internal_randen -labsl_random_internal_randen_slow -labsl_random_seed_sequences -labsl_random_internal_platform -labsl_random_internal_randen_hwaes -labsl_random_internal_seed_material -labsl_random_distributions -labsl_random_internal_pool_urbg -labsl_random_internal_randen_hwaes_impl -labsl_random_internal_platform
ABSL_RAN_FLAGS_LINUX=-l:libabsl_random_internal_randen.a
# ABSL_STAT_FLAGS_LINUX=-labsl_status -labsl_statusor
ABSL_STAT_FLAGS_LINUX=-l:libabsl_statusor.a -l:libabsl_status.a
# ABSL_STRINGS_FLAGS_LINUX=-labsl_cord -labsl_str_format_internal -labsl_strings -labsl_strings_internal
ABSL_STRINGS_FLAGS_LINUX=-l:libabsl_str_format_internal.a -l:libabsl_strings.a -l:libabsl_strings_internal.a -l:libabsl_cord.a
# ABSL_SYNC_FLAGS_LINUX=-labsl_graphcycles_internal -labsl_synchronization
ABSL_SYNC_FLAGS_LINUX=-l:libabsl_synchronization.a -l:libabsl_graphcycles_internal.a
# ABSL_TIME_FLAGS_LINUX=-labsl_civil_time -labsl_time -labsl_time_zone
ABSL_TIME_FLAGS_LINUX= -l:libabsl_time.a -l:libabsl_time_zone.a
# ABSL_TYPES_FLAGS_LINUX=-labsl_bad_any_cast_impl -labsl_bad_optional_access -labsl_bad_variant_access
ABSL_TYPES_FLAGS_LINUX=-l:libabsl_bad_optional_access.a -l:libabsl_bad_variant_access.a

RE2_FLAGS_LINUX=-l:libre2.a
BORING_FLAGS_LINUX=-l:libssl.a -l:libcrypto.a
CARES_FLAGS_LINUX=-l:libcares.a

VBZ_LIBNAME=libvbz.a
VBZ_FLAGS_LINUX=-l:$(VBZ_LIBNAME)

#----Linux Flags----#

CXX11=-std=c++11

#----Debug Specific Flags----#
FLASH_TEST_DEBUG= $(CUDA_DEBUG) $(DEBUG) -maxrregcount 26

#----make objects for linux----#
	
#----make objects for simple (linux)----#

all: $(PROGNAME)

clean:
	rm -rf bin/lib include/minknow_api submodules/grpc/cmake/build submodules/vbz_compression/build ReadUntilClient.o $(PROGNAME); \
	cd submodules/hdf5; \
	git clean -fd

$(DIR)/submodules/grpc/cmake/build/grpc_cpp_plugin:
	mkdir -p bin/lib; \
	mkdir log; \
	cd submodules/grpc; \
	mkdir -p cmake/build; \
	cd cmake/build; \
	CFLAGS=-Wno-error LDFLAGS=-lrt cmake ../..; \
	make; \
	cd third_party/zlib; \
	ln -s ../../../../third_party/zlib include

$(DIR)/submodules/grpc/cmake/build/third_party/protobuf/protoc: $(DIR)/submodules/grpc/cmake/build/grpc_cpp_plugin

$(DIR)/submodules/grpc/cmake/build/third_party/zlib/libz.a: $(DIR)/submodules/grpc/cmake/build/grpc_cpp_plugin
	
%.pb.cc: $(DIR)/submodules/grpc/cmake/build/third_party/protobuf/protoc $(DIR)/submodules/minknow_api/proto/minknow_api/*.proto
	$(DIR)/submodules/grpc/cmake/build/third_party/protobuf/protoc -I $(DIR)/submodules/minknow_api/proto/ -I $(DIR)/submodules/grpc/third_party/protobuf/src/ --grpc_out=$(DIR)/include/ --plugin=protoc-gen-grpc=$(DIR)/submodules/grpc/cmake/build/grpc_cpp_plugin $(DIR)/submodules/minknow_api/proto/minknow_api/*.proto; \
	$(DIR)/submodules/grpc/cmake/build/third_party/protobuf/protoc -I $(DIR)/submodules/minknow_api/proto/ -I $(DIR)/submodules/grpc/third_party/protobuf/src/ --cpp_out=$(DIR)/include/ $(DIR)/submodules/minknow_api/proto/minknow_api/*.proto

$(HDF5_LIB_LINUX)/$(HDF5_LIBNAME): $(DIR)/submodules/grpc/cmake/build/third_party/zlib/libz.a
	cd submodules/hdf5; \
	./configure --enable-cxx --enable-fortran --with-zlib=$(DIR)/submodules/grpc/cmake/build/third_party/zlib; \
	make -j 8; \
	make install
	
$(VBZ_LIB_LINUX)/$(VBZ_LIBNAME): 
	cd submodules/vbz_compression; \
	mkdir build; \
	cd build; \
	cmake -D CMAKE_BUILD_TYPE=Release -D ENABLE_CONAN=OFF -D ENABLE_PERF_TESTING=OFF -D ENABLE_PYTHON=OFF ..; \
	make -j

%.pb.o : %.pb.cc
	$(CC) $(CXX11) -I$(INCLUDE) -I$(INCLUDE_GRPC_LINUX) -I$(INCLUDE_ABSEIL_LINUX) -I$(INCLUDE_GOOGLE_LINUX) -c $< -o $@

ReadUntilClient.o: ReadUntilClient.cpp ReadUntilClient.h algo_datatypes.h Connection.h
	$(CC) $(CXX11) -I$(INCLUDE) -I$(INCLUDE_GRPC_LINUX) -I$(INCLUDE_ABSEIL_LINUX) -I$(INCLUDE_GOOGLE_LINUX) -c ReadUntilClient.cpp 

$(SO_LIB_PATH)/libReadUntilClient.a: $(ONT_CLIENT_FILES) ReadUntilClient.o 
	ar rcs $(SO_LIB_PATH)/libReadUntilClient.a ReadUntilClient.o $(ONT_CLIENT_FILES)

maxwell: $(HDF5_LIB_LINUX)/$(HDF5_LIBNAME) $(VBZ_LIB_LINUX)/$(VBZ_LIBNAME) $(SO_LIB_PATH)/libReadUntilClient.a ont_simple_client.cu thread.o cuda_utils.h wqueue.h flash_dtw.cuh flash_dtw_utils.cuh 
	$(NVCC) -ccbin `which $(CC) | xargs dirname` $(CXX11) -DHDF5_SUPPORTED=1 -I$(INCLUDE) -I$(INCLUDE_HDF5_LINUX) ont_simple_client.cu thread.o -o maxwell -L$(SO_LIB_PATH) -lReadUntilClient -L$(GRPC_LIB_LINUX) $(GRPC_FLAGS_LINUX) -L$(PROTOBUF_LIB_LINUX) $(PROTO_FLAGS_LINUX) -L$(ABSL_STRINGS_LIB_LINUX) $(ABSL_STRINGS_FLAGS_LINUX) -L$(ABSL_NUM_LIB_LINUX) $(ABSL_NUM_FLAGS_LINUX) -L$(ABSL_SYNC_LIB_LINUX) $(ABSL_SYNC_FLAGS_LINUX) -L$(ABSL_DEBUG_LIB_LINUX) $(ABSL_DEBUG_FLAGS_LINUX) -L$(ABSL_HASH_LIB_LINUX) $(ABSL_HASH_FLAGS_LINUX) -L$(ABSL_BASE_LIB_LINUX) $(ABSL_BASE_FLAGS_LINUX) -L$(ABSL_TIME_LIB_LINUX) $(ABSL_TIME_FLAGS_LINUX) -L$(ABSL_STAT_LIB_LINUX) $(ABSL_STAT_FLAGS_LINUX) -L$(ABSL_TYPES_LIB_LINUX) $(ABSL_TYPES_FLAGS_LINUX) -L$(RE2_LIB_LINUX) $(RE2_FLAGS_LINUX) -L$(BORING_LIB_LINUX) $(BORING_FLAGS_LINUX) -L$(CARES_LIB_LINUX) $(CARES_FLAGS_LINUX) -L$(ZLIB_LIB_LINUX) $(ZLIB_FLAGS_LINUX) -lpthread -lstdc++ -lm -L$(HDF5_LIB_LINUX) $(HDF5_FLAGS_LINUX) -L$(VBZ_LIB_LINUX) $(VBZ_FLAGS_LINUX)
	
thread.o: thread.cpp thread.h
	$(NVCC) -Xcompiler -fPIC -I. -DHAVE_STRUCT_TIMESPEC thread.cpp -c

#----Windows----#

#----Include Directories----#

DLL_LIB_PATH=lib

INCLUDE_HDF5_WIN="C:\Program Files\HDF_Group\HDF5\1.10.4\include"
INCLUDE_BOOST_WIN="C:\Program Files\boost\boost_1_70_0"
INCLUDE_GRPC_WIN=grpc/include
INCLUDE_GOOGLE_WIN=grpc/third_party/protobuf/src
INCLUDE_PTHREAD_WIN=./pthread-win32

HDF5_LIB_WIN="C:\Program Files\HDF_Group\HDF5\1.10.4\lib"
GRPC_LIB_WIN=grpc/bin/grpc/release
PROTO_LIB_WIN=grpc/bin/protobuf/release
EAY_LIB_WIN=grpc/bin/dependencies
ZLIB_LIB_WIN=grpc/bin/zlib/release/lib
WINDOW_LIB="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64"

PTHREAD_LIB=lib

#----Include Flags----#

HDF5_FLAGS_WIN=-lhdf5
GRPC_FLAGS_WIN=-lgpr -lgrpc++ -lgrpc
PROTO_FLAGS_WIN=-llibprotoc -llibprotobuf
EAY_FLAGS_WIN=-llibeay32 -lssleay32
ZLIB_FLAGS_WIN=-lzlib
WINDOW_FLAGS=-lWSock32 -lWS2_32 -lGdi32 -lUser32

PTHREAD_FLAGS=-lpthread_dll

#----make objects for simple (windows)----#

ont_simple_client.exe: ont_simple_client.cu thread.obj cuda_utils.h wqueue.h libmagenta_utils.dll ReadUntilClient.cuh Connection.h flash_dtw.cuh flash_dtw_utils.cuh
	$(NVCC) -I. -I$(INCLUDE) -I$(INCLUDE_HDF5_WIN) -I$(INCLUDE_PTHREAD_WIN) -I$(INCLUDE_GRPC_WIN) -I$(INCLUDE_GOOGLE_WIN) --expt-relaxed-constexpr -D_WIN32_WINNT=0x0600 $(ONT_CLIENT_FILES) thread.obj ont_simple_client.cu -o ont_simple_client.exe -L$(PTHREAD_LIB) $(PTHREAD_FLAGS) -L$(GRPC_LIB_WIN) $(GRPC_FLAGS_WIN) -L$(PROTO_LIB_WIN) $(PROTO_FLAGS_WIN) -L$(EAY_LIB_WIN) $(EAY_FLAGS_WIN) -L$(ZLIB_LIB_WIN) $(ZLIB_FLAGS_WIN) -L$(WINDOW_LIB) $(WINDOW_FLAGS) -L$(HDF5_LIB_WIN) $(HDF5_FLAGS_WIN) -L$(DLL_LIB_PATH) -llibmagenta_utils

check_bulk_data.exe: check_bulk_data.cu libmagenta_utils.dll
	$(NVCC) $(DEBUG) -I$(INCLUDE_HDF5_WIN) check_bulk_data.cu -o check_bulk_data.exe -L$(DLL_LIB_PATH) -llibmagenta_utils -L$(HDF5_LIB_WIN) $(HDF5_FLAGS_WIN)

thread.obj: thread.cpp thread.h
	$(NVCC) -I. -I$(INCLUDE_PTHREAD_WIN) -DHAVE_STRUCT_TIMESPEC thread.cpp -c

magenta.exe: magenta.cu libmagenta_utils.dll flash_dtw.cuh flash_utils.hpp cuda_utils.h
	$(NVCC) $(DEBUG) $(CUDA_DEBUG) -I. -I$(INCLUDE) -I$(INCLUDE_HDF5_WIN) magenta.cu -o magenta.exe -L$(HDF5_LIB_WIN) $(HDF5_FLAGS_WIN) -L$(DLL_LIB_PATH) -llibmagenta_utils