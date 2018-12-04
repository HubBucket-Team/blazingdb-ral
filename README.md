# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Requirements
- C++11 compiler
- CMake 3.11+
- Boost libs

# Dependencies
- cudf/libgdf development branch from BlazingDB fork https://github.com/BlazingDB/cudf/tree/develop/libgdf
- blazingdb-protocol/cpp development branch from https://github.com/BlazingDB/blazingdb-protocol/tree/develop/cpp
- Google Tests

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
Before build always update the gitsumbdule
```bash
cd blazingdb-ral
git submodule update --init --recursive
```

There are two ways to build the RAL component (for both cases you don't need to have conda in your system).

The first one will automagically download all the RAL dependencies as part of the cmake process.

```bash
cd blazingdb-ral
mkdir build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

The second one will reuse your development environment.
So you just need to pass cmake arguments for nvstrings, cudf/libgdf and blazingdb-protocol/cpp installation paths.  

```bash
cd blazingdb-ral
mkdir build
CUDACXX=/usr/local/cuda-9.2/bin/nvcc cmake -DCMAKE_BUILD_TYPE=Debug \
      -DFLATBUFFERS_INSTALL_DIR=/path/to/flatbuffers_install/ \
      -DARROW_INSTALL_DIR=/path/to/arrow_install/ \
      -DNVSTRINGS_INSTALL_DIR=/path/to/nvstrings_install/ \
      -DLIBGDF_INSTALL_DIR=/path/to/cudf-libgdf_install/ \
      -DBLAZINGDB_PROTOCOL_INSTALL_DIR=/path/to/blazingdb-protocol-cpp_install/ \
      -DBLAZINGDB_IO_INSTALL_DIR=/path/to/blazingdb-io_install/ \
      -DGOOGLETEST_INSTALL_DIR=/path/to/googletest_install/ \
      -DAWS_SDK_CPP_BUILD_DIR=/path/to/aws-sdk-cpp/build/
      ..
make
```

Remember NVSTRINGS_INSTALL_DIR and LIBGDF_INSTALL_DIR always got together.

If you don't define these optional arguments then the cmake process will resolve (download & build) each dependency:
- FLATBUFFERS_INSTALL_DIR
- ARROW_INSTALL_DIR
- BLAZINGDB_PROTOCOL_INSTALL_DIR
- BLAZINGDB_IO_INSTALL_DIR
- GOOGLETEST_INSTALL_DIR
- AWS_SDK_CPP_BUILD_DIR

Finally, if don't want to use conda and need the nvstrings library, just download https://anaconda.org/nvidia/nvstrings/0.0.3/download/linux-64/nvstrings-0.0.3-cuda9.2_py35_0.tar.bz2 and uncompress the folder, this folder is the NVSTRINGS_HOME.

# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```
