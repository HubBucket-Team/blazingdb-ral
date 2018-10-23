#pragma once

#include <cuda_runtime.h>

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
 
#define __DRIVER_TYPES_H__ 1 

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

static const char *_cudaGetErrorEnum(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";

    case cudaErrorMissingConfiguration:
      return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
      return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
      return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
      return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
      return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
      return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
      return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
      return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
      return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
      return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
      return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
      return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
      return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
      return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
      return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
      return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
      return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
      return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
      return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
      return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
      return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
      return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
      return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
      return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
      return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
      return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
      return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
      return "cudaErrorProfilerAlreadyStopped";

    /* Since CUDA 4.0*/
    case cudaErrorAssert:
      return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";

    /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
      return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
      return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
      return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
      return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
      return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
      return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
      return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";

    /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
      return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
      return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
      return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
      return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
      return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";

    /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
      return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
      return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";

    /* Since CUDA 8.0*/
    case cudaErrorNvlinkUncorrectable:
      return "cudaErrorNvlinkUncorrectable";

    /* Since CUDA 8.5*/
    case cudaErrorJitCompilerNotFound:
      return "cudaErrorJitCompilerNotFound";

    /* Since CUDA 9.0*/
    case cudaErrorCooperativeLaunchTooLarge:
      return "cudaErrorCooperativeLaunchTooLarge";
  }

  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}
#define CheckCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    DEVICE_RESET
    exit(EXIT_FAILURE);
  }
}

namespace libgdf {

static std::basic_string<int8_t> BuildCudaIpcMemHandler (void *data) {
  std::basic_string<int8_t> bytes;
  if (data != nullptr) {
    cudaIpcMemHandle_t ipc_memhandle;
    CheckCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &ipc_memhandle, (void *) data));

    std::basic_string<int8_t> bytes;
    bytes.resize(sizeof(cudaIpcMemHandle_t));
    memcpy((void*)bytes.data(), (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  
  }
  return bytes;
}


static void* CudaIpcMemHandlerFrom (const std::basic_string<int8_t>& handler) {
  void * response = nullptr;
  std::cout << "handler-content: " <<  handler.size() <<  std::endl;
  if (handler.size() == 64) {
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    CheckCudaErrors(cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess));
  }
  return response;
}

std::tuple<std::vector<std::vector<gdf_column_cpp>>,
           std::vector<std::string>,
           std::vector<std::vector<std::string>>> toBlazingDataframe(const ::blazingdb::protocol::TableGroupDTO& request,std::vector<void *> & handles)
{
  std::vector<std::vector<gdf_column_cpp>> input_tables;
  std::vector<std::string> table_names;
  std::vector<std::vector<std::string>> column_names;

  for(auto table : request.tables) {
    table_names.push_back(table.name);
    column_names.push_back(table.columnNames);

    std::vector<gdf_column_cpp> input_table;
    for(auto column : table.columns) {

    	gdf_column_cpp col = gdf_column_cpp(libgdf::CudaIpcMemHandlerFrom(column.data), (gdf_valid_type*)libgdf::CudaIpcMemHandlerFrom(column.valid), (::gdf_dtype)column.dtype, (size_t)column.size, (gdf_size_type)column.null_count);
    	    	handles.push_back( col.data());
    	    	handles.push_back((void *) col.valid());
    	    	input_table.push_back(col);
    }
    input_tables.push_back(input_table);
  }

  return std::make_tuple(input_tables, table_names, column_names);
}

} //namespace libgdf
