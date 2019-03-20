#include "Traits/RuntimeTraits.h"
#include "GDFColumn.cuh"
#include <cudf.h>

namespace ral {
namespace traits {

    gdf_size_type get_dtype_size_in_bytes(const gdf_column* column) {
        return gdf_dtype_size(column->dtype);
    }

    gdf_size_type get_dtype_size_in_bytes(gdf_dtype dtype) {
        return gdf_dtype_size(dtype);
    }

    gdf_size_type get_data_size_in_bytes(const gdf_column* column) {
        return (column->size * get_dtype_size_in_bytes(column->dtype));
    }

    gdf_size_type get_data_size_in_bytes(const gdf_column_cpp& column) {
        return (column.size()) * get_dtype_size_in_bytes(column.dtype());
    }

    gdf_size_type get_data_size_in_bytes(gdf_size_type quantity, gdf_dtype dtype) {
        return (quantity * get_dtype_size_in_bytes(dtype));
    }

    gdf_size_type get_bitmask_size_in_bytes(const gdf_column* column) {
        return gdf_valid_allocation_size(column->size);
    }

    gdf_size_type get_bitmask_size_in_bytes(gdf_size_type quantity) {
        return gdf_valid_allocation_size(quantity);
    }

} // namespace traits
} // namespace ral
