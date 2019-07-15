#pragma once

#include <string>
#include "GDFColumn.cuh"
#include "gdf_wrapper/gdf_types.cuh"

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

        using NvStrings = NVStrings;
        using NvCategory = NVCategory;

    public:
        static void copyGpuToCpu(std::size_t& binary_pointer, std::string& result, gdf_column_cpp& column);

        static std::size_t getDataCapacity(gdf_column* column);

        static std::size_t getValidCapacity(gdf_column* column);

        static std::size_t getDTypeSize(gdf_dtype dtype);

        static std::size_t isGdfString(const gdf_column &);

        static NVStrings * CreateNvStrings(const void *      stringsPointer,
                                           const void *      offsetsPointer,
                                           const std::size_t keysLength);
    };

} // namespace adapter
} // namespace communication
} // namespace ral
