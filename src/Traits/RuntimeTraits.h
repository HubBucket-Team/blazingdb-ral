#ifndef TRAITS_RUNTIME_TRAITS
#define TRAITS_RUNTIME_TRAITS

#include <cstdint>
#include <cstddef>
#include <cudf/types.h>

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

class gdf_column_cpp;

namespace ral {
namespace traits {

    constexpr std::size_t BYTE_SIZE_IN_BITS = 8;

    constexpr std::size_t BITMASK_SIZE_IN_BYTES = 64;


    gdf_size_type get_dtype_size_in_bytes(const gdf_column* column);

    gdf_size_type get_dtype_size_in_bytes(gdf_dtype dtype);


    gdf_size_type get_data_size_in_bytes(const gdf_column_cpp& column);

    gdf_size_type get_data_size_in_bytes(const gdf_column* column);

    gdf_size_type get_data_size_in_bytes(gdf_size_type quantity, gdf_dtype dtype);


    gdf_size_type get_bitmask_size_in_bytes(const gdf_column* column);

    gdf_size_type get_bitmask_size_in_bytes(gdf_size_type quantity);


    template <typename Type>
    struct GetDType;

    template<>
    struct GetDType<gdf_index_type> {
        constexpr static gdf_dtype dtype{GDF_INT32};
    };

    template <typename Type>
    constexpr gdf_dtype dtype = GetDType<Type>::dtype;

} // namespace traits
} // namespace ral

#endif
