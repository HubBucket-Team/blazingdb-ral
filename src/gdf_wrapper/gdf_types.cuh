#ifndef _BZ_OPS_H_
#define _BZ_OPS_H_

#include <cudf.h>

/**
 * @enum gdf_binary_operator
 * It contains the different operations that can be performed in the binary operations.
 * The enumeration is used in the following functions:
 * - gdf_binary_operation_v_s_v
 * - gdf_binary_operation_v_v_s
 * - gdf_binary_operation_v_v_v
 * - gdf_binary_operation_v_s_v_d
 * - gdf_binary_operation_v_v_s_d
 * - gdf_binary_operation_v_v_v_d
 */
typedef enum {
    GDF_ADD,
    GDF_SUB,
    GDF_MUL,
    GDF_DIV,
    GDF_TRUE_DIV,
    GDF_FLOOR_DIV,
    GDF_MOD,
    GDF_POW,
    GDF_EQUAL,
    GDF_NOT_EQUAL,
    GDF_LESS,
    GDF_GREATER,
    GDF_LESS_EQUAL,
    GDF_GREATER_EQUAL,
   	GDF_INVALID_BINARY
    //GDF_COMBINE,
    //GDF_COMBINE_FIRST,
    //GDF_ROUND,
    //GDF_PRODUCT,
    //GDF_DOT
} gdf_binary_operator;

typedef enum{
	GDF_FLOOR,
	GDF_CEIL,
	GDF_SIN,
	GDF_COS,
	GDF_ASIN,
	GDF_ACOS,
	GDF_TAN,
	GDF_COTAN,
	GDF_ATAN,
	GDF_ABS,
	GDF_NOT,
	GDF_LN,
	GDF_LOG,
	GDF_YEAR,
	GDF_MONTH,
	GDF_DAY,
	GDF_HOUR,
	GDF_MINUTE,
	GDF_SECOND,
	GDF_INVALID_UNARY

} gdf_unary_operator;

typedef enum{
    GDF_COALESCE
} gdf_other_binary_operator;


/**
 * @union gdf_data
 * @brief Union used for scalar type.
 * It stores a unique value for scalar type.
 * It has a direct relationship with the gdf_dtype.
 */
typedef union {
    int8_t   si08;  /**< GDF_INT8      */
    int16_t  si16;  /**< GDF_INT16     */
    int32_t  si32;  /**< GDF_INT32     */
    int64_t  si64;  /**< GDF_INT64     */
    uint8_t  ui08;  /**< GDF_UINT8     */
    uint16_t ui16;  /**< GDF_UINT16    */
    uint32_t ui32;  /**< GDF_UINT32    */
    uint64_t ui64;  /**< GDF_UINT64    */
    float    fp32;  /**< GDF_FLOAT32   */
    double   fp64;  /**< GDF_FLOAT64   */
    int32_t  dt32;  /**< GDF_DATE32    */
    int64_t  dt64;  /**< GDF_DATE64    */
    int64_t  tmst;  /**< GDF_TIMESTAMP */
} gdf_data;

/**
 * @struct gdf_scalar
 * @brief  literal or variable
 *
 * The struct is used as a literal or a variable in the libgdf library.
 *
 * @var data     A union that represents the value.
 * @var dtype    An enum that represents the type of the value.
 * @var is_valid A boolean that represents whether the scalar is null.
 */
typedef struct {
    gdf_data  data;
    gdf_dtype dtype;
    bool      is_valid;
} gdf_scalar;

/**
 * @brief Binary operation function between gdf_scalar and gdf_column structs.
 *
 * The function performs the binary operation of a gdf_scalar operand and a
 * gdf_column operand.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when the two
 * operands (vax and vay) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_scalar and
 * gdf_column structs due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_scalar) First operand of the operation.
 * @param vay (gdf_column) Second operand of the operation.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return    GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *            error code
 */
gdf_error gdf_binary_operation_v_s_v(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope);

/**
 * @brief Binary operation function between gdf_column and gdf_scalar structs.
 *
 * The function performs the binary operation of a gdf_column operand and a
 * gdf_scalar operand.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when the two
 * operands (vax and vay) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_scalar and
 * gdf_column structs due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_column) First operand of the operation.
 * @param vay (gdf_scalar) Second operand of the operation.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return    GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *            error code
 */
gdf_error gdf_binary_operation_v_v_s(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope);

/**
 * @brief Binary operation function between two gdf_column structs.
 *
 * The function performs the binary operation of two gdf_column operands.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when the two
 * operands (vax and vay) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_column
 * struct due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_column) First operand of the operation.
 * @param vay (gdf_column) Second operand of the operation.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return    GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *            error code
 */
gdf_error gdf_binary_operation_v_v_v(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope);

/**
 * @brief Binary operation function between gdf_scalar and gdf_column structs.
 *
 * The function performs the binary operation of a gdf_scalar operand and a
 * gdf_column operand. A default scalar operand is used to replace an operand
 * in the binary operation when such operand is null.
 *
 * Whether any operand (vax or vay) is null, then it will be replaced with the
 * default scalar operand value (def). In case both operands (vax and vay) are
 * null, each of them will be replaced with the default scalar operand value.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when two or three
 * operands (vax, vay and def) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_scalar and
 * gdf_column structs due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_column) First operand of the operation.
 * @param vay (gdf_scalar) Second operand of the operation - gdf_scalar.
 * @param def (gdf_scalar) Default operand used to replace a null operand.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return     GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *             error code
 */
gdf_error gdf_binary_operation_v_s_v_d(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope);

/**
 * @brief Binary operation function between gdf_column and gdf_scalar structs.
 *
 * The function performs the binary operation of a gdf_column operand and a
 * gdf_scalar operand. A default scalar operand is used to replace an operand
 * in the binary operation when such operand is null.
 *
 * Whether any operand (vax or vay) is null, then it will be replaced with the
 * default scalar operand value (def). In case both operands (vax and vay) are
 * null, each of them will be replaced with the default scalar operand value.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when two or three
 * operands (vax, vay and def) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_scalar and
 * gdf_column structs due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_column) First operand of the operation.
 * @param vay (gdf_scalar) Second operand of the operation - gdf_scalar.
 * @param def (gdf_scalar) Default operand used to replace a null operand.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return     GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *             error code
 */
gdf_error gdf_binary_operation_v_v_s_d(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope);

/**
 * @brief Binary operation function between two gdf_column structs.
 *
 * The function performs the binary operation of two gdf_column operands. A default
 * scalar operand is used to replace an operand in the binary operation when such
 * operand is null.
 *
 * Whether any gdf_column (vax or vay) is null, then it will be replaced with the
 * default scalar operand value (def). In case both operands (vax and vay) are null,
 * each of them will be replaced with the default scalar operand value.
 *
 * The valid field in the gdf_column output will be 1 (by bit) when two or three
 * operands (vax, vay and def) are not null. Otherwise, it will be 0 (by bit).
 *
 * It is required to set in an appropriate manner the fields in the gdf_scalar and
 * gdf_column structs due to that the binary operation will not be performed.
 *
 * @param out (gdf_column) Output of the operation.
 * @param vax (gdf_column) First operand of the operation.
 * @param vay (gdf_column) Second operand of the operation.
 * @param def (gdf_scalar) Default operand used to replace a null operand.
 * @param ope (enum) The binary operator that is going to be used in the operation.
 * @return    GDF_SUCCESS if the operation was successful, otherwise an appropriate
 *            error code.
 */
gdf_error gdf_binary_operation_v_v_v_d(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope);

#endif /* _BZ_OPS_H_ */
