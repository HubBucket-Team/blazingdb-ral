# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Dependencies
See https://github.com/BlazingDB/blazingdb-toolchain

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

```bash
cd blazingdb-ral
mkdir build
CUDACXX=/usr/local/cuda-9.2/bin/nvcc cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_TESTING=ON \
      -DBLAZINGDB_DEPENDENCIES_INSTALL_DIR=/foo/blazingsql/dependencies/ \
      -DLIBGDF_INSTALL_DIR=/foo/cudf_install_dir/ \
      -DBLAZINGDB_PROTOCOL_INSTALL_DIR=/foor/blazingdb_protocol_install_dir/ \
      -DBLAZINGDB_IO_INSTALL_DIR=/foo/blazingdb_io_install_dir/ \
      -DBLAZINGDB_COMMUNICATION_INSTALL_DIR=/foo/blazingdb_communication_install_dir/ \
      ..
make -j8
```

**NOTE:**
If you want to build the dependencies using the old C++ ABI, add this cmake argument:

```bash
-DCXX_OLD_ABI=ON
```

Also, If you want to show verbose, add these args to cmake:

```bash
-DCUDA_DEFINES=-DVERBOSE -DCXX_DEFINES=-DVERBOSE
```

# Integration Tests

```bash
./integration_test-gen.sh
mkdir -p  build && cd build
LIBGDF_HOME="/path/to/libgdf"cmake ..
make -j8
```
