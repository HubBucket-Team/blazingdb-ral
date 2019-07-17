#include "Traits/RuntimeTraits.h"
#include "GDFColumn.cuh"
#include <cudf.h>
#include <utilities/column_utils.hpp>

namespace ral {
namespace traits {

    gdf_size_type get_dtype_size_in_bytes(const gdf_column* column) {
        return cudf::size_of(column->dtype);        
    }

    gdf_size_type get_dtype_size_in_bytes(gdf_dtype dtype) {
        return cudf::size_of(dtype);        
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

    gdf_dtype convert_string_dtype(std::string str){
        if(str == "GDF_INT8"){
            return GDF_INT8;
        }else if(str == "GDF_INT16"){
            return GDF_INT16;
        }else if(str == "GDF_INT32"){
            return GDF_INT32;
        }else if(str == "GDF_INT64"){
            return GDF_INT64;
        }else if(str == "GDF_FLOAT32"){
            return GDF_FLOAT32;
        }else if(str == "GDF_FLOAT64"){
            return GDF_FLOAT64;
        }else if (str == "GDF_BOOL8"){
            return GDF_BOOL8;
        }else if(str == "GDF_DATE32"){
            return GDF_DATE32;
        }else if(str == "GDF_DATE64"){
            return GDF_DATE64;
        }else if(str == "GDF_TIMESTAMP"){
            return GDF_TIMESTAMP;
        }else if(str == "GDF_CATEGORY"){
            return GDF_CATEGORY;
        }else if(str == "GDF_STRING"){
            return GDF_STRING;
        }else if(str == "GDF_STRING_CATEGORY"){
            return GDF_STRING_CATEGORY;
        }else{
            return GDF_invalid;
        }
    }

} // namespace traits
} // namespace ral
