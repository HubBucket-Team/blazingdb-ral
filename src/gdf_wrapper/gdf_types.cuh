#ifndef _BZ_OPS_H_
#define _BZ_OPS_H_

#include <cudf.h>


// typedef enum {
//     GDF_ADD,
//     GDF_SUB,
//     GDF_MUL,
//     GDF_DIV,
//     GDF_TRUE_DIV,
//     GDF_FLOOR_DIV,
//     GDF_MOD,
//     GDF_POW,
//     GDF_EQUAL,
//     GDF_NOT_EQUAL,
//     GDF_LESS,
//     GDF_GREATER,
//     GDF_LESS_EQUAL,
//     GDF_GREATER_EQUAL,
//    	GDF_INVALID_BINARY,
//     GDF_COALESCE // a new one!
//     //GDF_COMBINE,
//     //GDF_COMBINE_FIRST,
//     //GDF_ROUND,
//     //GDF_PRODUCT,
//     //GDF_DOT
// } gdf_binary_operator;

typedef enum{
	BLZ_FLOOR,
	BLZ_CEIL,
	BLZ_SIN,
	BLZ_COS,
	BLZ_ASIN,
	BLZ_ACOS,
	BLZ_TAN,
	BLZ_COTAN,
	BLZ_ATAN,
	BLZ_ABS,
	BLZ_NOT,
	BLZ_LN,
	BLZ_LOG,
	BLZ_YEAR,
	BLZ_MONTH,
	BLZ_DAY,
	BLZ_HOUR,
	BLZ_MINUTE,
	BLZ_SECOND,
	BLZ_INVALID_UNARY

} gdf_unary_operator;

// /**
//  * @union gdf_data
//  * @brief Union used for scalar type.
//  * It stores a unique value for scalar type.
//  * It has a direct relationship with the gdf_dtype.
//  */
// typedef union {
//     int8_t   si08;  /**< GDF_INT8      */
//     int16_t  si16;  /**< GDF_INT16     */
//     int32_t  si32;  /**< GDF_INT32     */
//     int64_t  si64;  /**< GDF_INT64     */
//     uint8_t  ui08;  /**< GDF_UINT8     */
//     uint16_t ui16;  /**< GDF_UINT16    */
//     uint32_t ui32;  /**< GDF_UINT32    */
//     uint64_t ui64;  /**< GDF_UINT64    */
//     float    fp32;  /**< GDF_FLOAT32   */
//     double   fp64;  /**< GDF_FLOAT64   */
//     int32_t  dt32;  /**< GDF_DATE32    */
//     int64_t  dt64;  /**< GDF_DATE64    */
//     int64_t  tmst;  /**< GDF_TIMESTAMP */
// } gdf_data;

// /**
//  * @struct gdf_scalar
//  * @brief  literal or variable
//  *
//  * The struct is used as a literal or a variable in the libgdf library.
//  *
//  * @var data     A union that represents the value.
//  * @var dtype    An enum that represents the type of the value.
//  * @var is_valid A boolean that represents whether the scalar is null.
//  */
// typedef struct {
//     gdf_data  data;
//     gdf_dtype dtype;
//     bool      is_valid;
// } gdf_scalar;

#endif /* _BZ_OPS_H_ */
