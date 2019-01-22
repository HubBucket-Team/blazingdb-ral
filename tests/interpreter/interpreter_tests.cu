
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "Interpreter/interpreter_cpp.h"
#include "Interpreter/interpreter_ops.cuh"

#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <blazingdb/io/Util/StringUtil.h>
#include <gtest/gtest.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
//#include <Utils.cuh>

#include "gdf/library/scalar.h"
#include "gdf/library/table.h"
#include "gdf/library/table_group.h"
#include "gdf/library/types.h"
#include "gdf/library/api.h"
using namespace gdf::library;

struct EvaluateQueryTest : public ::testing::Test
{
    struct InputTestItem
    {
        std::string query;
        std::string logicalPlan;
        gdf::library::TableGroup tableGroup;
        gdf::library::Table resultTable;
    };

    void CHECK_RESULT(gdf::library::Table &computed_solution,
                      gdf::library::Table &reference_solution)
    {

        computed_solution.print(std::cout);
        reference_solution.print(std::cout);

        for (size_t index = 0; index < reference_solution.size(); index++)
        {
            const auto &reference_column = reference_solution[index];
            const auto &computed_column = computed_solution[index];
            auto a = reference_column.to_string();
            auto b = computed_column.to_string();
            EXPECT_EQ(a, b);
        }
    }
};


//  + * + $0 $1 $2 $1 , + $1 2
TEST_F(EvaluateQueryTest, TEST_00)
{

    Table input_table = TableBuilder{
        "emps",
        {
            {"x", [](Index i) -> DType<GDF_INT32> { return i * 2.0; }},
            {"y", [](Index i) -> DType<GDF_INT32> { return i * 10.0; }},
            {"z", [](Index i) -> DType<GDF_INT32> { return i * 20.0; }},
        }}
                            .Build(10);
    Table output_table = TableBuilder{
        "emps",
        {
            {"o1", [](Index i) -> DType<GDF_INT32> { return 0; }},
            {"o2", [](Index i) -> DType<GDF_INT32> { return 0; }},
        }}
                             .Build(10);

    input_table.print(std::cout);
    output_table.print(std::cout);

    std::vector<gdf_column_cpp> input_columns_cpp = input_table.ToGdfColumnCpps();
    std::vector<gdf_column_cpp> output_columns_cpp = output_table.ToGdfColumnCpps();

    std::vector<gdf_column *> output_columns(2);
    output_columns[0] = output_columns_cpp[0].get_gdf_column();
    output_columns[1] = output_columns_cpp[1].get_gdf_column();

    std::vector<gdf_column *> input_columns(3);
    input_columns[0] = input_columns_cpp[0].get_gdf_column();
    input_columns[1] = input_columns_cpp[1].get_gdf_column();
    input_columns[2] = input_columns_cpp[2].get_gdf_column();

    //step 0:   + * + $0 $1 $2 $1 , + $1 2
    //step 0:                expr1, expr2

    //step 1:  + * $5 $2 $1 , + $1 $2

    //step 2:  + $5 $1 , + $1 $2

    // Registers are
    // 	0			        1     				2			      3     			  4				      5	    		    6				        n + 3 + 2
    // input_col_1, input_col_2, input_col_3, output_col_1, output_col2, processing_1, processing_2 .... processing_n

    // expr1,    expr2:
    std::vector<column_index_type> left_inputs =  {0, 5, 5,      1};
    std::vector<column_index_type> right_inputs = {1, 2, 1,     -2};
    std::vector<column_index_type> outputs =      {5, 5, 3,      4};

    std::vector<column_index_type> final_output_positions = {3, 4};

    std::vector<gdf_binary_operator> operators = {GDF_ADD, GDF_MUL, GDF_ADD, GDF_ADD};
    std::vector<gdf_unary_operator> unary_operators = {GDF_INVALID_UNARY, GDF_INVALID_UNARY, GDF_INVALID_UNARY, GDF_INVALID_UNARY};

    using I32 = gdf::library::GdfEnumType<GDF_INT32>;

    gdf::library::Scalar<I32> junk_obj;
    junk_obj.setValue(0).setValid(true);
    gdf::library::Scalar<I32> vscalar_obj;
    vscalar_obj.setValue(2).setValid(true);

    gdf_scalar junk = *junk_obj.scalar();
    gdf_scalar scalar_val = *vscalar_obj.scalar();

    std::vector<gdf_scalar> left_scalars = {junk, junk, junk, junk, junk};
    std::vector<gdf_scalar> right_scalars = {junk, junk, junk, junk, junk};

    std::vector<column_index_type> new_input_indices = {0, 1, 2};

    auto error = perform_operation(output_columns, input_columns, left_inputs, right_inputs, outputs, final_output_positions, operators, unary_operators, left_scalars, right_scalars, new_input_indices);
    ASSERT_EQ(error, GDF_SUCCESS);
}


//  + * + $0 $1 $2 $1 , + sin $1 2.33
TEST_F(EvaluateQueryTest, TEST_01)
{

    Table input_table = TableBuilder{
        "emps",
        {
            {"x", [](Index i) -> DType<GDF_INT32> { return (i + 1) * 2.0; }},
            {"y", [](Index i) -> DType<GDF_INT32> { return (i + 1) * 10.0; }},
            {"z", [](Index i) -> DType<GDF_INT32> { return (i + 1) * 20.0; }},
        }}
                            .Build(10);
    Table output_table = TableBuilder{
        "emps",
        {
            {"o1", [](Index i) -> DType<GDF_INT32> { return 0; }},
            {"o2", [](Index i) -> DType<GDF_FLOAT32> { return 0; }},
        }}
                             .Build(10);

    input_table.print(std::cout);
    output_table.print(std::cout);

    std::vector<gdf_column_cpp> input_columns_cpp = input_table.ToGdfColumnCpps();
    std::vector<gdf_column_cpp> output_columns_cpp = output_table.ToGdfColumnCpps();

    std::vector<gdf_column *> output_columns(2);
    output_columns[0] = output_columns_cpp[0].get_gdf_column();
    output_columns[1] = output_columns_cpp[1].get_gdf_column();

    std::vector<gdf_column *> input_columns(3);
    input_columns[0] = input_columns_cpp[0].get_gdf_column();
    input_columns[1] = input_columns_cpp[1].get_gdf_column();
    input_columns[2] = input_columns_cpp[2].get_gdf_column();

    //step 0:   + * + $0 $1 $2 $1 , + sin $1 2.33
    //step 0:                expr1, expr2

    //step 1:  + * $5 $2 $1 , + $1 $2

    //step 2:  + $5 $1 , + $1 $2

    // Registers are
    // 	0			        1     				2			      3     			  4				      5	    		    6				        n + 3 + 2
    // input_col_1, input_col_2, input_col_3, output_col_1, output_col2, processing_1, processing_2 .... processing_n

    // expr1,    expr2:
    std::vector<column_index_type> left_inputs =  {0, 5, 5,      1, 5};
    std::vector<column_index_type> right_inputs = {1, 2, 1,     -1, -2};
    std::vector<column_index_type> outputs =      {5, 5, 3,      5, 4};

    std::vector<column_index_type> final_output_positions = {3, 4};

    std::vector<gdf_binary_operator> operators = {GDF_ADD, GDF_MUL, GDF_ADD, GDF_INVALID_BINARY, GDF_ADD};
    std::vector<gdf_unary_operator> unary_operators = { GDF_INVALID_UNARY,GDF_INVALID_UNARY,GDF_INVALID_UNARY,GDF_SIN,GDF_INVALID_UNARY };

    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;

    gdf::library::Scalar<FP32> junk_obj;
    junk_obj.setValue(0.0).setValid(true);
    gdf::library::Scalar<FP32> vscalar_obj;
    vscalar_obj.setValue(2.33).setValid(true);

    gdf_scalar junk = *junk_obj.scalar();
    gdf_scalar scalar_val = *vscalar_obj.scalar();

    std::vector<gdf_scalar> left_scalars = {junk, junk, junk, junk, junk};
    std::vector<gdf_scalar> right_scalars = {junk, junk, junk, scalar_val, junk};

    std::vector<column_index_type> new_input_indices = {0, 1, 2};

    auto error = perform_operation(output_columns, input_columns, left_inputs, right_inputs, outputs, final_output_positions, operators, unary_operators, left_scalars, right_scalars, new_input_indices);
    ASSERT_EQ(error, GDF_SUCCESS);
}
