
/* --------------------------------------------------------------------------*
 * @brief This function is a binary function. It will take in two gdf_columns.
 * The first one is expected to be a regular gdf_column, the second one
 * has to be a column of the same type as the first, and it has to be of
 * size one or of the same size as the other column.
 * 
 * case 1: If the second column contains only one value, then this funciton will
 * replace all nulls in the first column with the value in the second
 * column.
 *  
 * case 2: If the second column is of the same size as the first, then the function will
 * replace all nulls of the first column with the corresponding elemetns of the
 * second column
 * 
 * @Param[out] first gdf_column
 * @Param[in] second gdf_column, new_values_column column
 * 
 * @Returns GDF_SUCCESS upon successful completion
 *
 * --------------------------------------------------------------------------*/
#include <cudf.h>

#include "replace.cuh"

#include "../../utilities/error_utils.h"
#include "../../utilities/type_dispatcher.hpp"

constexpr int BLOCK_SIZE = 256;

template <typename Type>
__global__
void replace_nulls_with_scalar_kernel(int size, Type* va_out_data, Type* vax_in_data, uint32_t* vax_in_valid, const Type *vay_data_scalar) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    int index = i / warpSize;
    uint32_t position = i % warpSize;
    uint32_t is_vax_valid = vax_in_valid[index];

    uint32_t sel_vax = (is_vax_valid >> position) & 1;
    va_out_data[i] = sel_vax? vax_in_data[i] : *vay_data_scalar;
  }
}


template <typename Type>
__global__
void replace_nulls_with_column_kernel(int size, Type* va_out_data,  Type* vax_data, uint32_t* vax_valid, const Type *vay_data) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    int index = i / warpSize;
    uint32_t position = i % warpSize;
    uint32_t is_vax_valid = vax_valid[index];

    uint32_t sel_vax = (is_vax_valid >> position) & 1;
    va_out_data[i] = sel_vax? vax_data[i] : vay_data[i];
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the apropiate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_kernel_forwarder {
  template <typename col_type>
  void operator()(size_t           nrows,
                  size_t           new_values_length,
                  void*            d_col_out_data,
                  void*            d_col_in_data,
                  gdf_valid_type*  d_col_in_valid,
                  const void*      d_new_value)
  {
    const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
    if (new_values_length == 1) {
      replace_nulls_with_scalar_kernel<<<grid_size, BLOCK_SIZE>>>(nrows,
                                            static_cast<col_type*>(d_col_out_data),
                                            static_cast<col_type*>(d_col_in_data),
                                            (uint32_t*)d_col_in_valid,
                                            static_cast<const col_type*>(d_new_value)
                                            );
    } else if(new_values_length == nrows) {
      replace_nulls_with_column_kernel<<<grid_size, BLOCK_SIZE>>>(nrows,
                                            static_cast<col_type*>(d_col_out_data),
                                            static_cast<col_type*>(d_col_in_data),
                                            (uint32_t*)(d_col_in_valid),
                                            static_cast<const col_type*>(d_new_value)
                                            );
      
    }
  }
};

/* --------------------------------------------------------------------------*
 * @brief This function is a binary function. It will take in two gdf_columns.
 * The first one is expected to be a regular gdf_column, the second one
 * has to be a column of the same type as the first, and it has to be of
 * size one or of the same size as the other column.
 * 
 * case 1: If the second column contains only one value, then this funciton will
 * replace all nulls in the first column with the value in the second
 * column.
 *  
 * case 2: If the second column is of the same size as the first, then the function will
 * replace all nulls of the first column with the corresponding elemetns of the
 * second column
 * 
 * @Param[out] first gdf_column
 * @Param[in] second gdf_column, new_values_column column
 * 
 * @Returns GDF_SUCCESS upon successful completion
 *
 * --------------------------------------------------------------------------*/
gdf_error gdf_replace_nulls(gdf_column* col_out, gdf_column* col_in, const gdf_column* new_values_column)
{
  GDF_REQUIRE(col_out->dtype == new_values_column->dtype, GDF_DTYPE_MISMATCH);
  GDF_REQUIRE(new_values_column->size == 1 || new_values_column->size == col_out->size, GDF_COLUMN_SIZE_MISMATCH);

  GDF_REQUIRE(col_in->size == col_out->size, GDF_COLUMN_SIZE_MISMATCH);
    
  cudf::type_dispatcher(col_out->dtype, replace_nulls_kernel_forwarder{},
                          col_out->size,
                          new_values_column->size,
                          col_out->data,
                          col_in->data,
                          col_in->valid,
                          new_values_column->data);
  return GDF_SUCCESS;
}
 