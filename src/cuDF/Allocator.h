#pragma once

#include <string>
#include <cuda_runtime_api.h>

namespace cuDF {
namespace Allocator {

void allocate(void** pointer, std::size_t size, cudaStream_t stream = 0);

void reallocate(void** pointer, std::size_t size, cudaStream_t stream = 0);

void deallocate(void* pointer, cudaStream_t stream = 0);


class Exception {
public:
    virtual ~Exception()
    { }

    virtual const char* what() const noexcept = 0;
};

class Message : public Exception {
public:
    Message(std::string&& message);

public:
    const char* what() const noexcept override;

private:
    const std::string message;
};

class CudaError : public Message {
public:
    CudaError();
};

class InvalidArgument : public Message {
public:
    InvalidArgument();
};

class NotInitialized : public Message {
public:
    NotInitialized();
};

class OutOfMemory : public Message {
public:
    OutOfMemory();
};

class InputOutput : public Message {
public:
    InputOutput();
};

class Unknown : public Message {
public:
    Unknown();
};

} // namespace Allocator
} // namespace cuDF
