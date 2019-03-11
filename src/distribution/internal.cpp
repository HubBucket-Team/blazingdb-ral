#include "internal.h"

#include <unordered_map>

#include <cuda_runtime.h>

namespace internal {

gdf_column_cpp
slice(const gdf_column_cpp &col,
      const gdf_size_type   start,
      const gdf_size_type   length) {
    cudaError_t cudaStatus;
    gdf_column *gdf_col = col.get_gdf_column();

    const std::unordered_map<gdf_dtype, gdf_size_type> DTypeSizeOf{
      {GDF_invalid, -1},
      {GDF_INT8, 1},
      {GDF_INT16, 2},
      {GDF_INT32, 4},
      {GDF_INT64, 8},
      {GDF_FLOAT32, 4},
      {GDF_FLOAT64, 8},
      {GDF_DATE32, 4},
      {GDF_DATE64, 8},
      {GDF_TIMESTAMP, 8},
    };

    const gdf_size_type dtypeSize = DTypeSizeOf[gdf_col->dtype];
    const gdf_size_type dataSize  = length * dtypeSize;

    void *data = nullptr;
    cudaStatus = cudaMalloc(&data, static_cast<std::size_t>(dataSize));
    if (cudaSuccess != cudaStatus) {
        throw std::runtime_error("Slice cudaMalloc");
    }

    cudaStatus = cudaMemcpy(data,
                            gdf_col->data + (start * dtypeSize),
                            dataSize,
                            cudaMemcpyDeviceToDevice);

    gdf_valid_type *valid = nullptr;

    const gdf_size_type size = length;

    const gdf_size_type null_count = 0;

    char *col_name = nullptr;

    gdf_column *new_gdf_col = new gdf_column{data,
                                             valid,
                                             size,
                                             gdf_col->dtype,
                                             null_count,
                                             gdf_col->dtype_info,
                                             col_name};

    gdf_column_cpp result;
    result.create_gdf_column(new_gdf_col);
    return result;
}

}  // namespace internal
