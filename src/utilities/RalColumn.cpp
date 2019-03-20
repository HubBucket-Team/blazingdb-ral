#include "utilities/RalColumn.h"

namespace ral {
namespace utilities {

gdf_column_cpp create_zero_column(const gdf_size_type size, const gdf_dtype dtype, std::string&& name) {
    return create_zero_column(size, dtype, name);
}

gdf_column_cpp create_zero_column(const gdf_size_type size, const gdf_dtype dtype, const std::string& name) {
    // create data array
    std::size_t data_size = ral::traits::get_data_size_in_bytes(size, dtype);
    std::vector<std::uint8_t> data(data_size, 0);

    // create bitmask array
    std::size_t bitmask_size = ral::traits::get_bitmask_size_in_bytes(size);
    std::vector<std::uint8_t> bitmask(bitmask_size, 0);

    // create gdf_column_cpp
    gdf_column_cpp column;
    auto width = ral::traits::get_dtype_size_in_bytes(dtype);
    column.create_gdf_column(dtype, size, data.data(), bitmask.data(), width, name);

    // done
    return column;
}

} // namespace utilities
} // namespace ral
