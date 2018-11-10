#ifndef TRAITS_RUNTIME_TRAITS
#define TRAITS_RUNTIME_TRAITS

#include <gdf/gdf.h>

namespace Ral {
namespace Traits {

inline bool is_dtype_float32(gdf_dtype type) {
    return type == GDF_FLOAT32;
}

inline bool is_dtype_float64(gdf_dtype type) {
    return type == GDF_FLOAT64;
}

inline bool is_dtype_float(gdf_dtype type) {
    return (type == GDF_FLOAT32) || (type == GDF_FLOAT64);
}

inline bool is_dtype_signed(gdf_dtype type) {
    return (type == GDF_INT8    ||
            type == GDF_INT16   ||
            type == GDF_INT32   ||
            type == GDF_INT64   ||
            type == GDF_FLOAT32 ||
            type == GDF_FLOAT64);
}

inline bool is_dtype_unsigned(gdf_dtype type) {
    return (type == GDF_UINT8  ||
            type == GDF_UINT16 ||
            type == GDF_UINT32 ||
            type == GDF_UINT64);
}

inline bool is_dtype_integer(gdf_dtype type) {
    return (type == GDF_INT8   ||
            type == GDF_UINT8  ||
            type == GDF_INT16  ||
            type == GDF_UINT16 ||
            type == GDF_INT32  ||
            type == GDF_UINT32 ||
            type == GDF_INT64  ||
            type == GDF_UINT64);
}

} // Traits
} // Ral

#endif
