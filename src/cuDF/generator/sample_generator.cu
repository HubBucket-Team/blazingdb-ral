#include "cuDF/generator/sample_generator.h"
#include "cuDF/generator/random_generator.cuh"

namespace cudf {
namespace generator {

    gdf_error generate_sample(std::vector<gdf_column_cpp>& data_frame,
                              std::vector<gdf_column_cpp>& sampled_data,
                              gdf_size_type num_samples) {
        cudf::generator::RandomVectorGenerator<unsigned int> generator(0L, data_frame.size());
        auto array = generator(num_samples);

        for (std::size_t k = 0; k < num_samples; ++k) {
            sampled_data.emplace_back(data_frame[array[k]].clone());
        }

        return GDF_SUCCESS;
    }

} // namespace generator
} // namespace cudf
