#pragma once

#include "GDFColumn.cuh"

namespace ral {
namespace test {

gdf_column_cpp create_gdf_column_cpp(std::size_t size, gdf_dtype dtype);

gdf_column_cpp create_null_gdf_column_cpp(std::size_t size, gdf_dtype dtype);

std::vector<std::uint8_t> get_column_data(gdf_column* column);

std::vector<std::uint8_t> get_column_valid(gdf_column* column);

bool operator==(const gdf_column& lhs, const gdf_column& rhs);

bool operator==(const gdf_column_cpp& lhs, const gdf_column_cpp& rhs);

} // namespace test
} // namespace ral
