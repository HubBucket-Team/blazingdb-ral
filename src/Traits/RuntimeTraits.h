#ifndef TRAITS_RUNTIME_TRAITS
#define TRAITS_RUNTIME_TRAITS

#include "gdf_wrapper/gdf_wrapper.cuh"

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

//TODO felipe percy noboa see upgrade to uints
//inline bool is_dtype_unsigned(gdf_dtype type) {
//    return (type == GDF_UINT8  ||
//            type == GDF_UINT16 ||
//            type == GDF_UINT32 ||
//            type == GDF_UINT64);
//}

inline bool is_dtype_integer(gdf_dtype type) {
    return (type == GDF_INT8   ||
//            type == GDF_UINT8  ||
            type == GDF_INT16  ||
//            type == GDF_UINT16 ||
            type == GDF_INT32  ||
//            type == GDF_UINT32 ||
            type == GDF_INT64//  ||
//            type == GDF_UINT64
			);
}

} // Traits
} // Ral

namespace ral {
namespace traits {
    constexpr std::size_t BYTE_SIZE_IN_BITS = 8;

    constexpr std::size_t VALID_SIZE_IN_BYTES = 64;


    std::size_t get_dtype_size(const gdf_column* column);

    std::size_t get_dtype_size(gdf_dtype dtype);


    std::size_t get_data_size(const gdf_column* column);

    std::size_t get_data_size(std::size_t quantity, gdf_dtype dtype);


    std::size_t get_valid_size(const gdf_column* column);

    std::size_t get_valid_size(std::size_t quantity);

} // namespace traits
} // namespace ral

#endif
