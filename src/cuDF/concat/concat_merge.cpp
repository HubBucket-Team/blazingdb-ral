#include <algorithm>
#include <gdf_wrapper/gdf_wrapper.cuh>
#include "cuDF/concat/concat_merge.h"

namespace cudf {
namespace concat {

    gdf_error concat_merge(std::vector<blazing_frame>& input_frames,
                           blazing_frame& output_frame,
                           bool destroy_inputs) {

        if (input_frames.size() == 0) {
            return GDF_DATASET_EMPTY;
        }

        // verify whether all the gdf_column have the same dtype

        auto& input_frame = input_frames[0].get_columns();

        output_frame.get_columns().resize(1);
        output_frame.get_columns()[0].resize(input_frame.size());

        std::transform(input_frame.begin(),
                       input_frame.end(),
                       output_frame.get_columns()[0].begin(),
                       [destroy_inputs](std::vector<gdf_column_cpp>& input_vector) -> gdf_column_cpp {

                           std::size_t total_size_concat = 0;
                           std::for_each(input_vector.begin(),
                                         input_vector.end(),
                                         [&total_size_concat](gdf_column_cpp& column) {
                                             total_size_concat += column.size();
                                         });

                           gdf_column_cpp output_column;
                           auto& input = input_vector[0];

                           int width;
                           auto error = get_column_byte_width(input.get_gdf_column(), &width);
                           if (error != GDF_SUCCESS) {
                               //exception: GDF_UNSUPPORTED_DTYPE
                               return gdf_column_cpp();
                           }
                           output_column.create_gdf_column(input.dtype(),
                                                           total_size_concat,
                                                           nullptr,
                                                           width,
                                                           "concat-" + input.name());

                           std::vector<gdf_column*> columns;
                           std::transform(input_vector.begin(),
                                          input_vector.end(),
                                          std::back_inserter(columns),
                                          [](gdf_column_cpp& input) {
                                              return input.get_gdf_column();
                                          });

                           error = gdf_column_concat(output_column.get_gdf_column(), columns.data(), columns.size());

                           if (destroy_inputs) {
                               std::for_each(input_vector.begin(),
                                             input_vector.end(),
                                             [](gdf_column_cpp& input) {
                                                 input = gdf_column_cpp();
                                             });
                           }
                           //if (destroy_inputs) {
                           //    input_vector.clear();
                           //}

                           return output_column;
                       });

        return GDF_SUCCESS;
    }

} // namespace concat
} // namespace cudf
