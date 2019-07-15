#include "cuDF/generator/sample_generator.h"
#include "cuDF/generator/random_generator.cuh"
#include "CalciteExpressionParsing.h"
#include <copying.hpp>
#include <types.hpp>
#include "table.hpp"

namespace cudf {
namespace generator {

gdf_error generate_sample(std::vector<gdf_column_cpp>& data_frame,
                          std::vector<gdf_column_cpp>& sampled_data,
                          gdf_size_type num_samples) {
    if (data_frame.size() == 0) {
        return GDF_DATASET_EMPTY;
    }
    
    if (num_samples <= 0) {
        sampled_data = data_frame;
        return GDF_SUCCESS;
    }

    cudf::generator::RandomVectorGenerator<int32_t> generator(0L, data_frame[0].size());
    std::vector<int32_t> arrayIdx = generator(num_samples);

    // Gather
    sampled_data.clear();
    sampled_data.resize(data_frame.size());
    std::vector<gdf_column*> raw_data_frame(data_frame.size());
    std::vector<gdf_column*> raw_sampled_data(data_frame.size());
    for(size_t i = 0; i < data_frame.size(); i++) {
        sampled_data[i].create_gdf_column(data_frame[i].dtype(),
                                        arrayIdx.size(),
                                        nullptr,
                                        get_width_dtype(data_frame[i].dtype()),
                                        data_frame[i].name());
        raw_sampled_data[i] = sampled_data[i].get_gdf_column();
        raw_data_frame[i] = data_frame[i].get_gdf_column();
    }

    cudf::table srcTable{raw_data_frame.data(), (gdf_size_type)raw_data_frame.size()};
    cudf::table destTable{raw_sampled_data.data(), (gdf_size_type)raw_sampled_data.size()};

    gdf_column_cpp gatherMap;
    gatherMap.create_gdf_column(GDF_INT32, arrayIdx.size(), arrayIdx.data(), get_width_dtype(GDF_INT32), "");

    // std::cout << "Gather Map\n";
    // print_gdf_column(gatherMap.get_gdf_column());

    cudf::gather(&srcTable, (gdf_index_type*)(gatherMap.get_gdf_column()->data), &destTable);

    return GDF_SUCCESS;
}

} // namespace generator
} // namespace cudf
