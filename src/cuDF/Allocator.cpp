#include "cuDF/Allocator.h"
#include "rmm/rmm.h"

#include "../FreeMemory.h"

namespace cuDF {
namespace Allocator {

const std::string BASE_MESSAGE {"ERROR, cuDF::Allocator, "};

void throwException(rmmError_t error);

void allocate(void** pointer, std::size_t size, cudaStream_t stream) {

    cudaMalloc(pointer, size);

    // auto error = RMM_ALLOC(pointer, size, stream);
    // FreeMemory::registerRawPointer(*pointer);
    // if (error != RMM_SUCCESS) {
    //     throwException(error);
    // }
}

void reallocate(void **pointer, std::size_t size, cudaStream_t stream) {
    
    const void *actual = *pointer;
    auto error = RMM_REALLOC(pointer, size, stream);
    FreeMemory::updateRawPointer(actual, *pointer);
    if (error != RMM_SUCCESS) {
        throwException(error);
    }
}

void deallocate(void* pointer, cudaStream_t stream) {

    cudaFree(pointer);

    // auto error = RMM_FREE(pointer, stream);
    // FreeMemory::removeRawPointer(pointer);
    // if (error != RMM_SUCCESS) {
    //     throwException(error);
    // }
}

void throwException(rmmError_t error) {
    switch(error) {
    case RMM_ERROR_CUDA_ERROR:
        throw CudaError();
    case RMM_ERROR_INVALID_ARGUMENT:
        throw InvalidArgument();
    case RMM_ERROR_NOT_INITIALIZED:
        throw NotInitialized();
    case RMM_ERROR_OUT_OF_MEMORY:
        throw OutOfMemory();
    case RMM_ERROR_UNKNOWN:
        throw Unknown();
    case RMM_ERROR_IO:
        throw InputOutput();
    default:
        throw Unknown();
    }
}


Message::Message(std::string&& message)
 : message{message}
{ }

const char* Message::what() const noexcept {
    return message.c_str();
}

CudaError::CudaError()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_CUDA_ERROR:" +
           std::to_string(RMM_ERROR_CUDA_ERROR) +
           ", Error in CUDA")
{ }

InvalidArgument::InvalidArgument()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_INVALID_ARGUMENT:" +
           std::to_string(RMM_ERROR_INVALID_ARGUMENT) +
           ", Invalid argument was passed")
{ }

NotInitialized::NotInitialized()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_NOT_INITIALIZED:" +
           std::to_string(RMM_ERROR_NOT_INITIALIZED) +
           ", RMM API called before rmmInitialize()")
{ }

OutOfMemory::OutOfMemory()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_OUT_OF_MEMORY:" +
           std::to_string(RMM_ERROR_OUT_OF_MEMORY) +
           ", Unable to allocate more memory")
{ }

Unknown::Unknown()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_UNKNOWN:" +
           std::to_string(RMM_ERROR_UNKNOWN) +
           ", Unknown error")
{ }

InputOutput::InputOutput()
 : Message(BASE_MESSAGE +
           "RMM_ERROR_IO:" +
           std::to_string(RMM_ERROR_IO) +
           ", Stats output error")
{ }

} // namespace Allocator
} // namespace cuDF
