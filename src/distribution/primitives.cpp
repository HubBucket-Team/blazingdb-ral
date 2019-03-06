#include "distribution/primitives.h"


namespace cudf {
namespace generator {

gdf_error generate_sample(std::vector<gdf_column_cpp>& data_frame,
        std::vector<gdf_column_cpp>& sampled_data,
        gdf_size_type num_samples) {
    return GDF_SUCCESS;
}

} // namespace generator
} // namespace cudf


namespace ral {
namespace distribution {

auto generateSamples(std::vector<std::vector<gdf_column_cpp>>& input_tables,
                     std::vector<std::size_t>& quantities)
    -> std::vector<std::vector<gdf_column_cpp>> {
    // verify
    if (input_tables.size() != quantities.size()) {
        throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__} + " -- size mismatch.");
    }
    // output data
    std::vector<std::vector<gdf_column_cpp>> result;

    // make sample for each table
    for (std::size_t k = 0; k < input_tables.size(); ++k) {
        std::vector<gdf_column_cpp> sample;
        auto error = cudf::generator::generate_sample(input_tables[k], sample, quantities[k]);
        if (error != GDF_SUCCESS) {
            throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__} + " -- CUDF: " + gdf_error_get_name(error));
        }
        result.emplace_back(std::move(sample));
    }

    // done
    return result;
}

} // namespace distribution
} // namespace ral