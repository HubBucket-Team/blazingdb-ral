# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Dependencies
- C++11 compiler
- CMake 3.11+
- LibGDF (branch: binary-operators-draft)
- FlatBuffers
- BlazingDB Procotol
- Google Test

Except for the compiler and CMake, all the dependencies will be downloaded and configured automatically.

# Building

```bash
mkdir build
cd build
cmake ..
make -j8
```

To avoid the libgdf building, make sure to set the LIBGDF_HOME environment variable to wherever you have statically built the libgdf repository in the branch binary-operators-draft.

To build LIBGDF statically:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DLIBGDF_STATIC_LIB=ON ..
```

Then, build the RAL with:
```bash
export LIBGDF_HOME="/path/to/libgdf"
cmake ..
make -j8
```

or with:
```bash
LIBGDF_HOME="/path/to/libgdf" cmake ..
make -j8
```
# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```