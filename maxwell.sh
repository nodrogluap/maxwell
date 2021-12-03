#/usr/bin/env bash

HDF5_PLUGIN_PATH=`dirname $0`/submodules/vbz_compression/build/bin `dirname $0`/maxwell $*
