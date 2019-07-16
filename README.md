# blazingdb-ral
BlazingDB Relational Algebra Interpreter

# Dependencies
- General dependencies: https://github.com/BlazingDB/blazingdb-toolchain
- BlazingDB Protocol library: https://github.com/BlazingDB/blazingdb-protocol
- BlazingDB IO library: https://github.com/BlazingDB/blazingdb-io
- BlazingDB Communication: https://github.com/BlazingDB/blazingdb-communication
- BlazingDB/cudf (develop branch): https://github.com/BlazingDB/cudf/tree/develop

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
      -DBLAZINGDB_PROTOCOL_INSTALL_DIR=/foo/blazingdb_protocol_install_dir/ \
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

# Run

```bash
./testing-libgdf RAL_ID GPU_ID ORCHESTRATOR_HTTP_COMMUNICATION_[IP|HOSTNAME] ORCHESTRATOR_HTTP_COMMUNICATION_PORT RAL_HTTP_COMMUNICATION_[IP|HOSTNAME] RAL_HTTP_COMMUNICATION_PORT RAL_TCP_PROTOCOL_PORT

#Example if running on one server with two GPU

#Instance 1 for localhost:
./testing-libgdf 1 0 127.0.0.1 9000 127.0.0.1 9001 8891

#Instance 2 for localhost:
./testing-libgdf 2 1 127.0.0.1 9000 127.0.0.1 9002 8892

```
