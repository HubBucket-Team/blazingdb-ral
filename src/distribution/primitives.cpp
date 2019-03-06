#include "distribution/primitives.h"
#include "cuDF/generator/sample_generator.h"

#include <cmath>

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

namespace sampling {

double
percentage(std::size_t tableSize) {
    return std::exp(-tableSize);
}

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, double percentage) {
    std::size_t num_samples = table[0].size() * percentage / 100;

    std::vector<gdf_column_cpp> sample;

    gdf_error gdf_status =
      cudf::generator::generate_sample(table, sample, num_samples);
    if (GDF_SUCCESS != gdf_status) {
        throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__}
                                 + " -- CUDF: "
                                 + gdf_error_get_name(gdf_status));
    }

    return sample;
}

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &tables,
                const std::vector<double> &               percentages) {
    // verify
    if (tables.size() != percentages.size()) {
        throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__}
                                 + " -- size mismatch.");
    }

    // output data
    std::vector<std::vector<gdf_column_cpp>> samples;
    samples.reserve(tables.size());

    // make sample for each table
    for (std::size_t i = 0; i <tables.size(); i ++) {
      samples.emplace_back(generateSample(tables[i], percentages[i]));
    }

    // done
    return samples;
}

}  // namespace sampling

} // namespace distribution
} // namespace ral
