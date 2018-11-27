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
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DNVSTRINGS_HOME=/path/to/nvstrings/home/install/ \
      -DLIBGDF_HOME=/path/to/cudf-libgdf/home/install/ \
      -DBLAZINGDB_PROTOCOL_HOME=/path/to/blazingdb-protocol-cpp/home/install/ \
      ..
make
```

Notes:
- NVSTRINGS_HOME and LIBGDF_HOME always got together.
- BLAZINGDB_PROTOCOL_HOME is optional (if you no pass this arg then the project will download a copy of blazingdb-protocol)
- If you pass BLAZINGDB_PROTOCOL_HOME but not NVSTRINGS_HOME and LIBGDF_HOME, then the project will download a copy of nvstrings lib and cudf/libgdf.
- If don't want to use conda and need the nvstrings library, just download https://anaconda.org/nvidia/nvstrings/0.0.3/download/linux-64/nvstrings-0.0.3-cuda9.2_py35_0.tar.bz2 and uncompress the folder, this folder is the NVSTRINGS_HOME.

# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```