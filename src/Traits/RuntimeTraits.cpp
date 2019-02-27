#include "Traits/RuntimeTraits.h"
#include <arrow/util/bit-util.h>
#include "gdf_wrapper/utilities/cudf_utils.h"

namespace ral {
namespace traits {

    std::size_t get_dtype_size(gdf_dtype dtype) {
        int width;
        gdf_column column;
        column.dtype = dtype;
        get_column_byte_width(&column, &width);
        return (std::size_t) width;
    }

    std::size_t get_dtype_size(gdf_column* column) {
        int width;
        get_column_byte_width(column, &width);
        return (std::size_t) width;
    }

    std::size_t get_data_size(gdf_column* column) {
        return (column->size * get_dtype_size(column->dtype));
    }

    std::size_t get_data_size(std::size_t size, gdf_dtype dtype) {
        return (size * get_dtype_size(dtype));
    }

    std::size_t get_valid_size(std::size_t data_size) {
        return (std::size_t)PaddedLength(arrow::BitUtil::BytesForBits(data_size));
    }

} // namespace traits
} // namespace ral
