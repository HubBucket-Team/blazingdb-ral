#include "communication/adapter/GpuFunctionsAdapter.h"
#include "Traits/RuntimeTraits.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace adapter {

    void GpuFunctionsAdapter::copyGpuToCpu(std::size_t& binary_pointer, std::string& result, gdf_column_cpp& column)
    {
        std::size_t data_size = getDataCapacity(column.get_gdf_column());
        cudaMemcpy(&result[binary_pointer], column.get_gdf_column()->data, data_size, cudaMemcpyDeviceToHost);
        binary_pointer += data_size;

        std::size_t valid_size = getValidCapacity(column.get_gdf_column());
        cudaMemcpy(&result[binary_pointer], column.get_gdf_column()->valid, valid_size, cudaMemcpyDeviceToHost);
        binary_pointer += valid_size;
    }

    std::size_t GpuFunctionsAdapter::getDataCapacity(gdf_column* column) {
        return ral::traits::get_data_size(column);
    }

    std::size_t GpuFunctionsAdapter::getValidCapacity(gdf_column* column) {
        return ral::traits::get_valid_size(column->size);
    }

    std::size_t GpuFunctionsAdapter::getDTypeSize(gdf_dtype dtype) {
        return ral::traits::get_dtype_size(dtype);
    }

} // namespace adapter
} // namespace communication
} // namespace ral
