#ifndef BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
#define BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H

#include <vector>
#include "GDFColumn.cuh"

namespace ral {
namespace distribution {

auto generateSamples(std::vector<std::vector<gdf_column_cpp>>& input_tables, std::vector<std::size_t>& quantities)
    -> std::vector<std::vector<gdf_column_cpp>>;

} // namespace distribution
} // namespace ral

#endif //BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
