# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Requirements
- g++-5.4
- CMake 3.11+
- Make

# Dependencies
- nvstrings
- boost
- aws_sdk_cpp
- flatbuffers
- lz4
- zstd
- brotli
- snappy
- thrift
- arrow
- libgdf (cudf/cpp)
- blazingdb-protocol
- blazingdb-io
- GoogleTest

# Build the dependencies
Setup your workspace and output folders:
```bash
mkdir workspace
mkdir output
cd workspace
wget https://github.com/BlazingDB/blazingdb-automation/blob/develop/docker/blazingsql-build/blazingsql-build.properties
```

The blazingsql-build.properties describes how you want to build BlazingSQL, if you want to build only the dependencies then disable the unnecessary modules:
```bash
...
#optional: enable build (default is true)
cudf_enable=true
blazingdb_protocol_enable=true
blazingdb_io_enable=true
blazingdb_ral_enable=false
blazingdb_orchestrator_enable=false
blazingdb_calcite_enable=false
pyblazing_enable=false
...
```

Finally run the build.sh script: 
```bash
wget https://github.com/BlazingDB/blazingdb-automation/blob/develop/docker/blazingsql-build/build.sh
chmod +x build.sh
./build.sh /path/to/workspace /path/to/output
```

All the dependencies will be inside /path/to/workspace/

# Clone
This repo uses submodules. Make sure you cloned recursively:

```bash
git clone --recurse-submodules git@github.com:BlazingDB/blazingdb-ral.git
```

Or, after cloning:

```bash
cd blazingdb-ral
git submodule update --init --recursive
```

# Build
There are two ways to build the RAL component (for both cases you don't need to have conda in your system).

## First approach: Basic build
The first one will automagically download all the RAL dependencies as part of the cmake process.

```bash
cd blazingdb-ral
mkdir build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Second approach: Custom build with dependencies
This second approach will reuse your development environment.
So you just need to pass cmake arguments for installation paths of the dependencies you want.

```bash
cd blazingdb-ral
mkdir build
CUDACXX=/usr/local/cuda-9.2/bin/nvcc cmake -DCMAKE_BUILD_TYPE=Debug \
      -DNVSTRINGS_INSTALL_DIR=/path/to/workspace/dependencies/nvstrings_install_dir \
      -DBOOST_INSTALL_DIR=/path/to/workspace/dependencies/boost_install_dir \
      -DAWS_SDK_CPP_BUILD_DIR=/path/to/workspace/dependencies/aws_sdk_cpp_build_dir \
      -DFLATBUFFERS_INSTALL_DIR=/path/to/workspace/dependencies/flatbuffers_install_dir \
      -DLZ4_INSTALL_DIR=/path/to/workspace/dependencies/lz4_install_dir \
      -DZSTD_INSTALL_DIR=/path/to/workspace/dependencies/zstd_install_dir \
      -DBROTLI_INSTALL_DIR=/path/to/workspace/dependencies/brotli_install_dir \
      -DSNAPPY_INSTALL_DIR=/path/to/workspace/dependencies/snappy_install_dir \
      -DTHRIFT_INSTALL_DIR=/path/to/workspace/dependencies/thrift_install_dir \
      -DARROW_INSTALL_DIR=/path/to/workspace/dependencies/_install_dir \
      -DLIBGDF_INSTALL_DIR=/path/to/workspace/dependencies/libgdf_install_dir \
      -DBLAZINGDB_PROTOCOL_INSTALL_DIR=/path/to/workspace/blazingdb-protocol_project/$branch/install \
      -DBLAZINGDB_IO_INSTALL_DIR=/path/to/workspace/blazingdb-io_project/$branch/install \
      -DGOOGLETEST_INSTALL_DIR=/path/to/workspace/dependencies/googletest_install_dir \
      ..
make
```

Remember NVSTRINGS_INSTALL_DIR and LIBGDF_INSTALL_DIR always got together.

Also, if you don't define these optional arguments then the cmake process will resolve (download & build) each dependency:
- NVSTRINGS_INSTALL_DIR
- BOOST_INSTALL_DIR
- AWS_SDK_CPP_BUILD_DIR
- FLATBUFFERS_INSTALL_DIR
- LZ4_INSTALL_DIR
- ZSTD_INSTALL_DIR
- BROTLI_INSTALL_DIR
- SNAPPY_INSTALL_DIR
- THRIFT_INSTALL_DIR
- ARROW_INSTALL_DIR
- LIBGDF_INSTALL_DIR
- BLAZINGDB_PROTOCOL_INSTALL_DIR
- BLAZINGDB_IO_INSTALL_DIR
- GOOGLETEST_INSTALL_DIR

Finally, if don't want to use conda and need the nvstrings library, just download https://anaconda.org/nvidia/nvstrings/0.0.3/download/linux-64/nvstrings-0.0.3-cuda9.2_py35_0.tar.bz2 and uncompress the folder, this folder is the NVSTRINGS_HOME.

# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```