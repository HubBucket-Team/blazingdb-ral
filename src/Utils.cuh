#ifndef UTILS_CUH_
#define UTILS_CUH_

#include "gdf_wrapper/gdf_wrapper.cuh"

#include <iostream>
#include <vector>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <rmm.h>


#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

#define CUDF_CALL( call )                                            \
{                                                                    \
  gdf_error err = call;                                              \
  if ( err != GDF_SUCCESS )                                          \
  {                                                                  \
    std::cerr << "ERROR: CUDF Runtime call " << #call                \
              << " in line " << __LINE__                             \
              << " of file " << __FILE__                             \
              << " failed with " << gdf_error_get_name(err)          \
              << " (" << err << ").\n";                              \
    /* Call cudaGetLastError to try to clear error if the cuda context is not corrupted */ \
    cudaGetLastError();                                              \
    throw std::runtime_error("In " + std::string(#call) + " function: CUDF Runtime call error " + gdf_error_get_name(err));\
  }                                                               \
}

#define CheckCudaErrors( call )                                      \
{                                                                    \
  cudaError_t cudaStatus = call;                                     \
  if (cudaSuccess != cudaStatus)                                     \
  {                                                                  \
    std::cerr << "ERROR: CUDA Runtime call " << #call                \
              << " in line " << __LINE__                             \
              << " of file " << __FILE__                             \
              << " failed with " << cudaGetErrorString(cudaStatus)   \
              << " (" << cudaStatus << ").\n";                       \
    /* Call cudaGetLastError to try to clear error if the cuda context is not corrupted */ \
    cudaGetLastError();                                              \
    throw std::runtime_error("In " + std::string(#call) + " function: CUDA Runtime call error " + cudaGetErrorName(cudaStatus));\
  }                                                                  \
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
    cudaMemcpy((int *) h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  }

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

	return; // TODO alguien arregle esta funcion!!

//	HostDataType * host_data_out = new HostDataType[column->size];
//	char * host_valid_out;
//
//	if(column->size % GDF_VALID_BITSIZE != 0){
//		host_valid_out = new char[(column->size + (GDF_VALID_BITSIZE - (column->size % GDF_VALID_BITSIZE)))/GDF_VALID_BITSIZE];
//	}else{
//		host_valid_out = new char[column->size / GDF_VALID_BITSIZE];
//	}
//
//	int column_width;
//	get_column_byte_width(column, &column_width);
//
//	cudaMemcpy(host_data_out,column->data,column_width * column->size, cudaMemcpyDeviceToHost);
//	if (column->valid != nullptr)
//		cudaMemcpy(host_valid_out,column->valid,sizeof(gdf_valid_type) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
//	else
//		std::cout<<"Valid is null\n";
//
//	std::cout<<"Printing Column address ptr: "<<column<<", Size: "<<column->size<<"\n"<<std::flush;
//
//	for(int i = 0; i < column->size; i++){
//		int col_position = i / GDF_VALID_BITSIZE;
//		int bit_offset = GDF_VALID_BITSIZE - (i % GDF_VALID_BITSIZE);
//		std::cout<<"host_data_out["<<i<<"] = "<<(host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
//	}
//
//	delete[] host_data_out;
//	delete[] host_valid_out;
//
//	std::cout<<std::endl<<std::endl;
}

void free_gdf_column(gdf_column * column);

void gdf_sequence(int32_t* data, size_t size, int32_t init_val);

#endif /* UTILS_CUH_ */
