
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
#include <Utils.cuh>

#include "gdf/library/api.h"
using namespace gdf::library;

struct InputTestItem
{
    std::string query;
    std::string logicalPlan;
    gdf::library::TableGroup tableGroup;
};

struct EvaluateQueryTest : public ::testing::Test
{

    InputTestItem input;

    EvaluateQueryTest() : input{InputTestItem{
                              .query = "select id + salary from main.emps",
                              .logicalPlan =
                                  "LogicalProject(EXPR$0=[+($0, $2)])\n  "
                                  "EnumerableTableScan(table=[[main, emps]])",
                              .tableGroup =
                                  LiteralTableGroupBuilder{
                                      {"main.emps",
                                       {{"id", Literals<GDF_INT32>{Literals<GDF_INT32>::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}, Literals<GDF_INT32>::valid_vector{0xF0, 0x0F}}},
                                        {"age",  Literals<GDF_INT32>{Literals<GDF_INT32>::vector{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}, Literals<GDF_INT32>::valid_vector{0xF0, 0x0F}}},
                                        {"salary", Literals<GDF_INT32>{Literals<GDF_INT32>::vector{9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0}, Literals<GDF_INT32>::valid_vector{0xF0, 0x0F}}}
                                       }
                                       }}.Build()}}
    {
    }

    void create_input()
    {
        // LiteralTableGroupBuilder{
        //       {"main.emps",
        //         {"$0", Literals<GDF_INT32>{Literals<GDF_INT32>::vector{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}}},
        //         {"$1", Literals<GDF_INT32>{Literals<GDF_INT32>::vector{9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0}}}}}

        auto input_table = gdf::library::LiteralTableBuilder{"customer",
                                                             {
                                                                 gdf::library::LiteralColumnBuilder{
                                                                     "x",
                                                                     Literals<GDF_FLOAT32>{Literals<GDF_FLOAT32>::vector{1.1, 3.6, 5.3, 7.9, 9.35}},
                                                                 },
                                                                 gdf::library::LiteralColumnBuilder{
                                                                     "y",
                                                                     Literals<GDF_FLOAT32>{Literals<GDF_FLOAT32>::vector{1, 1, 1, 1, 1}},
                                                                 },
                                                                 gdf::library::LiteralColumnBuilder{
                                                                     "z",
                                                                     Literals<GDF_FLOAT32>{Literals<GDF_FLOAT32>::vector{0, 2, 4, 6, 8}},
                                                                 },
                                                             }} 
                               .Build();
    }

    void CHECK_RESULT(gdf::library::Table &computed_solution,
                      gdf::library::Table &reference_solution)
    {
        computed_solution.print(std::cout);
        reference_solution.print(std::cout);

        for (size_t index = 0; index < reference_solution.size(); index++)
        {
            const auto &reference_column = reference_solution[index];
            const auto &computed_column = computed_solution[index];

            // auto valid = ((int)computed_column.getValids()[0] & 1);
            // EXPECT_TRUE(valid == valid_result[index]);

            auto a = reference_column.to_string();
            auto b = computed_column.to_string();
            EXPECT_EQ(a, b);
        }
    }
};

TEST_F(EvaluateQueryTest, LiteralTableWithNulls)
{
    Literals<GDF_FLOAT64>::vector values = {7, 5, 6, 3, 1, 2, 8, 4, 1, 2, 8, 4, 1, 2, 8, 4};
    Literals<GDF_FLOAT64>::valid_vector valids = {0xF0, 0x0F};
    auto t = LiteralTableBuilder("emps",
                                 {
                                     LiteralColumnBuilder{
                                         "x",
                                         Literals<GDF_FLOAT64>{Literals<GDF_FLOAT64>::vector{values}, Literals<GDF_FLOAT64>::valid_vector{valids}},
                                     },
                                 })
                 .Build();
    t.print(std::cout);
    auto u = GdfColumnCppsTableBuilder{"emps", t.ToGdfColumnCpps()}.Build();

    for (std::size_t i = 0; i < 16; i++) {
        std::cout << t[0][i].get<GDF_FLOAT64>() << std::endl;
        EXPECT_EQ( values[i], t[0][i].get<GDF_FLOAT64>());
    }
    const auto &computed_valids = t[0].getValids();
    std::cout << "valids.size: " << valids.size() << std::endl;
    std::cout << "computed_valids.size: " << computed_valids.size() << std::endl;

    for (std::size_t i = 0; i < valids.size(); i++) {
        std::cout << "valids.size: " << valids[i] << std::endl;
        std::cout << "computed_valids.size: " << computed_valids[i] << std::endl;
        EXPECT_EQ(valids[i], computed_valids[i]); 
    }
    EXPECT_EQ(t, u);
}

// EnumerableTableScan(table=[[main, emps]])
TEST_F(EvaluateQueryTest, TEST_01)
{
    auto logical_plan = input.logicalPlan;
    auto input_tables = input.tableGroup.ToBlazingFrame();
    auto table_names = input.tableGroup.table_names();
    auto column_names = input.tableGroup.column_names();
    std::vector<gdf_column_cpp> outputs;

    blazing_frame bz_frame;
    for (auto &t : input_tables)
        bz_frame.add_table(t);

    std::vector<std::string> plan = StringUtil::split(logical_plan, "\n");

    auto params = parse_project_plan(bz_frame, plan[0]);
    gdf_error err = GDF_SUCCESS;

    //perform operations
    if (params.num_expressions_out > 0)
    {
        err = perform_operation(params.output_columns,
                                params.input_columns,
                                params.left_inputs,
                                params.right_inputs,
                                params.outputs,
                                params.final_output_positions,
                                params.operators,
                                params.unary_operators,
                                params.left_scalars,
                                params.right_scalars,
                                params.new_column_indices);
    }

    //params.output_columns
    std::vector<gdf_column_cpp> output_columns_cpp;

    for (gdf_column *col : params.output_columns)
    {
        gdf_column_cpp gdf_col;
        gdf_col.create_gdf_column(col);
        output_columns_cpp.push_back(gdf_col);
    }

    EXPECT_TRUE(err == GDF_SUCCESS);
    auto output_table = GdfColumnCppsTableBuilder{"output_table", output_columns_cpp}.Build();
    output_table.print(std::cout);

    err = evaluate_query(input_tables, table_names, column_names,
                         logical_plan, outputs);

    EXPECT_TRUE(err == GDF_SUCCESS);
    auto reference_table =
        GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
    CHECK_RESULT(output_table, reference_table);
}