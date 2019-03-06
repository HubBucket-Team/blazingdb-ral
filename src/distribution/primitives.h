#ifndef BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
#define BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H

#include <vector>
#include "GDFColumn.cuh"

namespace ral {
namespace distribution {

auto generateSamples(std::vector<std::vector<gdf_column_cpp>>& input_tables, std::vector<std::size_t>& quantities)
    -> std::vector<std::vector<gdf_column_cpp>>;


namespace sampling {

double
percentage(gdf_size_type tableSize);

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, double percentage);

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &tables,
                const std::vector<double> &               percentages);

void
prepareSamplesForGeneratePivots(
  std::vector<std::vector<gdf_column_cpp>> &samples,
  const std::vector<gdf_size_type> &        tableSizes);

}  // namespace sampling

} // namespace distribution
} // namespace ral

#endif //BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
