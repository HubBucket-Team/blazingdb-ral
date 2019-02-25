#pragma once

#include <string>
#include <exception>
#include <cuda_runtime_api.h>

namespace cuDF {
namespace Allocator {

void allocate(void** pointer, std::size_t size, cudaStream_t stream = 0);

void reallocate(void** pointer, std::size_t size, cudaStream_t stream = 0);

void deallocate(void* pointer, cudaStream_t stream = 0);


class CudfAllocatorError : public std::exception {
public:
    CudfAllocatorError(std::string&& message);
    const char* what() const noexcept override;
private:
    const std::string message;
};

class CudaError : public CudfAllocatorError {
public:
    CudaError();
};

class InvalidArgument : public CudfAllocatorError {
public:
    InvalidArgument();
};

class NotInitialized : public CudfAllocatorError {
public:
    NotInitialized();
};

class OutOfMemory : public CudfAllocatorError {
public:
    OutOfMemory();
};

class InputOutput : public CudfAllocatorError {
public:
    InputOutput();
};

class Unknown : public CudfAllocatorError {
public:
    Unknown();
};

} // namespace Allocator
} // namespace cuDF
