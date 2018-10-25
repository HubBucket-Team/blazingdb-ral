#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <iostream>
#include <vector>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
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

static constexpr int ValidSize = 32;
using ValidType = uint32_t;

static size_t  valid_size(size_t column_length)
{
  const size_t n_ints = (column_length / ValidSize) + ((column_length % ValidSize) ? 1 : 0);
  return n_ints * sizeof(ValidType);
}



static bool get_bit(const gdf_valid_type* const bits, size_t i)
{
  return  bits == nullptr? true :  bits[i >> size_t(3)] & (1 << (i & size_t(7)));
}


// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, 
                                                 std::function<void(gdf_column*)>>;

template <typename col_type>
void print_typed_column(col_type * col_data, 
                        gdf_valid_type * validity_mask, 
                        const size_t num_rows)
{

  std::vector<col_type> h_data(num_rows);
  cudaMemcpy(h_data.data(), col_data, num_rows * sizeof(col_type), cudaMemcpyDeviceToHost);


  const size_t num_masks = valid_size(num_rows);
  std::vector<gdf_valid_type> h_mask(num_masks);
  if(nullptr != validity_mask)
  {
    cudaMemcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  }
  for(size_t i = 0; i < num_rows; ++i)
  {
    if (sizeof(col_type) == 1)
      std::cout << (int)h_data[i] << " ";
    else
      std::cout << h_data[i] << " ";
  }
  std::cout << std::endl;
  return ;
  if (validity_mask == nullptr) {
    for(size_t i = 0; i < num_rows; ++i)
    {
      if (sizeof(col_type) == 1)
        std::cout << (int)h_data[i] << " ";
      else
        std::cout << h_data[i] << " ";
    }
  } 
  else {
    for(size_t i = 0; i < num_rows; ++i)
    {
        std::cout << "(" << std::to_string(h_data[i]) << "|" << get_bit(h_mask.data(), i) << "), ";
    }
  }
  std::cout << std::endl;
}

static void print_gdf_column(gdf_column const * the_column)
{
  const size_t num_rows = the_column->size;

  const gdf_dtype gdf_col_type = the_column->dtype;
  switch(gdf_col_type)
  {
    case GDF_INT8:
      {
        using col_type = int8_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT16:
      {
        using col_type = int16_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT32:
      {
        using col_type = int32_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT64:
      {
        using col_type = int64_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT32:
      {
        using col_type = float;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT64:
      {
        using col_type = double;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    default:
      {
        std::cout << "Attempted to print unsupported type.\n";
      }
  }
}


template <typename HostDataType>
void print_column(gdf_column * column){
	// @ todo : fix print column 
	return ;

	HostDataType * host_data_out = new HostDataType[column->size];
	char * host_valid_out;

	if(column->size % GDF_VALID_BITSIZE != 0){
		host_valid_out = new char[(column->size + (GDF_VALID_BITSIZE - (column->size % GDF_VALID_BITSIZE)))/GDF_VALID_BITSIZE];
	}else{
		host_valid_out = new char[column->size / GDF_VALID_BITSIZE];
	}

	int column_width;
	get_column_byte_width(column, &column_width);

	cudaMemcpy(host_data_out,column->data,column_width * column->size, cudaMemcpyDeviceToHost);
	if (column->valid != nullptr)
		cudaMemcpy(host_valid_out,column->valid,sizeof(gdf_valid_type) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
	else
		std::cout<<"Valid is null\n";

	std::cout<<"Printing Column address ptr: "<<column<<", Size: "<<column->size<<"\n"<<std::flush;

	for(int i = 0; i < column->size; i++){
		int col_position = i / GDF_VALID_BITSIZE;
		int bit_offset = GDF_VALID_BITSIZE - (i % GDF_VALID_BITSIZE);
		std::cout<<"host_data_out["<<i<<"] = "<<(host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
	}

	delete[] host_data_out;
	delete[] host_valid_out;

	std::cout<<std::endl<<std::endl;
}

void free_gdf_column(gdf_column * column);

#endif /* UTILS_CUH_ */
