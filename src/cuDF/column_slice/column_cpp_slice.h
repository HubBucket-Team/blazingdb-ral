#pragma once

#include "GDFColumn.cuh"

namespace ral {
namespace cudf {

gdf_error column_cpp_valid_slice(gdf_column_cpp*       output_column,
                                 const gdf_column_cpp* input_column,
                                 const gdf_size_type   start_bit,
                                 const gdf_size_type   bits_length);

} // namespace cudf
} // namespace ral
