#!/bin/bash

mkdir -p bin/lib
mkdir log
# PREFIX_PATH=`pwd`/bin/
CONFIGURE_PATH=`pwd`
# if [ "$1" != "" ]; then
	# PREFIX_PATH=$1
# fi
# echo "Libraries will be built to: ${PREFIX_PATH}"
echo "Building grpc libraries:"
cd submodules/grpc
mkdir -p cmake/build
cd cmake/build
# CFLAGS=-Wno-error CXXFLAGS=-lrt LDFLAGS=-lrt cmake ../..
CFLAGS=-Wno-error LDFLAGS=-lrt cmake ../..
make
cd third_party/zlib
ln -s ${CONFIGURE_PATH}/submodules/grpc/third_party/zlib include
echo "grpc libraries built!"
# echo "Building protobuf libraries"
# cd ../../third_party/protobuf/
# ./autogen.sh
# ./configure --prefix=`pwd`/../../../../bin
# make
# make check
# make install
# echo "protobuf libraries built!"
# cd ../../../../
cd ../../../../../../
echo "Building *.pb.cc andd *.pb.h files using *.proto files:"
# # Change output directories and bin 
PROTOC_PATH=${CONFIGURE_PATH}/submodules/grpc/cmake/build/third_party/protobuf
${PROTOC_PATH}/protoc -I submodules/minknow_api/proto/ -I submodules/grpc/third_party/protobuf/src/ --grpc_out=include/ --plugin=protoc-gen-grpc=submodules/grpc/cmake/build/grpc_cpp_plugin submodules/minknow_api/proto/minknow_api/*.proto
# Change output directories and bin
${PROTOC_PATH}/protoc -I submodules/minknow_api/proto/ -I submodules/grpc/third_party/protobuf/src/ --cpp_out=include/ submodules/minknow_api/proto/minknow_api/*.proto
echo "Build complete!"
echo "Building hdf5 libraries"
cd submodules/hdf5
./configure --enable-cxx --enable-fortran --with-zlib=${CONFIGURE_PATH}/submodules/grpc/cmake/build/third_party/zlib
make -j 8
make install
# make check
echo "Script finished"
