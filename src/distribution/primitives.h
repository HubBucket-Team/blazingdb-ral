#ifndef BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
#define BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H

#include <vector>
#include "GDFColumn.cuh"
#include "distribution/NodeColumns2.h"
#include "blazingdb/communication/Context.h"

namespace ral {
namespace distribution {
namespace sampling {

constexpr double thresholdForSubsampling = 0.01;

double
sampleRatio(gdf_size_type tableSize);

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, double ratio);

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &tables,
                const std::vector<double> &               ratios);

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, std::size_t quantity);

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &input_tables,
                std::vector<std::size_t> &                quantities);

void
prepareSamplesForGeneratePivots(
  std::vector<std::vector<gdf_column_cpp>> &samples,
  const std::vector<gdf_size_type> &        tableSizes);

}  // namespace sampling
}  // namespace distribution
}  // namespace ral

namespace ral {
namespace distribution {

namespace {
using Context = blazingdb::communication::Context;
} // namespace

void sendSamplesToMaster(const Context& context, std::vector<gdf_column_cpp>&& samples, std::size_t total_row_size);

std::vector<NodeColumns2> collectPartition(const Context& context);

}  // namespace distribution
}  // namespace ral

#endif  //BLAZINGDB_RAL_DISTRIBUTION_PRIMITIVES_H
