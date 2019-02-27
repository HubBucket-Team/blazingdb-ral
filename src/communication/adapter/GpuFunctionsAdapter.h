#pragma once

#include <string>
#include "gdf_wrapper/gdf_types.cuh"

class gdf_column_cpp;

namespace ral {
namespace communication {
namespace adapter {

    struct GpuFunctionsAdapter {
    public:
        using DType = gdf_dtype;
        using DTypeInfo = gdf_dtype_extra_info;
        using TimeUnit = gdf_time_unit;

        using DataTypePointer = void*;
        using ValidTypePointer = gdf_valid_type*;

    public:
        static void copyGpuToCpu(std::size_t& binary_pointer, std::string& result, gdf_column_cpp& column);

        static std::size_t getDataCapacity(gdf_column* column);

        static std::size_t getValidCapacity(gdf_column* column);

        static std::size_t getDTypeSize(gdf_dtype dtype);
    };

} // namespace adapter
} // namespace communication
} // namespace ral
