# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Dependencies
- C++11 compiler
- CMake 3.11+
- Boost libs

# Build
There are two ways to build the RAL component (for both cases you don't to have conda in your system).

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

# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```