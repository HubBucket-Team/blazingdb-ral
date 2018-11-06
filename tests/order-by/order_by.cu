
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <StringUtil.h>
#include <gdf/gdf.h>
#include <gtest/gtest.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include <Utils.cuh>

#include "gdf/library/api.h"
using namespace gdf::library;

struct EvaluateQueryTest : public ::testing::Test {
  struct InputTestItem {
    std::string query;
    std::string logicalPlan;
    gdf::library::TableGroup tableGroup;
    gdf::library::Table resultTable;
  };

  void CHECK_RESULT(gdf::library::Table& computed_solution,
                    gdf::library::Table& reference_solution) {
    computed_solution.print(std::cout);
    reference_solution.print(std::cout);

    for (size_t index = 0; index < reference_solution.size(); index++) {
      const auto& reference_column = reference_solution[index];
      const auto& computed_column = computed_solution[index];
      auto a = reference_column.to_string();
      auto b = computed_column.to_string();
      EXPECT_EQ(a, b);
    }
  }
};

// AUTO GENERATED UNIT TESTS
TEST_F(EvaluateQueryTest, DISABLED_TEST_00) { //Todo: not supported yet
  auto input = InputTestItem{
      .query = "select r_regionkey from main.region order by r_regionkey desc",
      .logicalPlan =
          "LogicalSort(sort0=[$0], dir0=[DESC])\n  "
          "LogicalProject(r_regionkey=[$0])\n    "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT8", Literals<GDF_INT8>{4, 3, 2, 1, 0}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_01) {
  auto input = InputTestItem{
      .query = "select r_regionkey from main.region order by r_regionkey",
      .logicalPlan =
          "LogicalSort(sort0=[$0], dir0=[ASC])\n  "
          "LogicalProject(r_regionkey=[$0])\n    "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT8", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_02) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_acctbal from main.customer order by c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$1], dir0=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_acctbal=[$1])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{37, 11, 33, 17, 2,  41, 47, 22, 1,  5,
                                    28, 35, 40, 21, 10, 15, 4,  23, 12, 32,
                                    48, 13, 50, 49, 16, 36, 26, 31, 14, 18,
                                    27, 46, 39, 38, 8,  25, 44, 3,  20, 29,
                                    6,  9,  34, 42, 19, 24, 30, 7,  43, 45}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    -917.75, -272.6,  -78.56,  6.34,    121.65,  270.95,
                    274.58,  591.98,  711.56,  794.47,  1007.18, 1228.24,
                    1335.3,  1428.25, 2753.54, 2788.52, 2866.83, 3332.02,
                    3396.49, 3471.53, 3792.5,  3857.34, 4266.13, 4573.94,
                    4681.03, 4987.27, 5182.05, 5236.89, 5266.3,  5494.43,
                    5679.84, 5744.59, 6264.31, 6345.11, 6819.74, 7133.7,
                    7315.94, 7498.12, 7603.4,  7618.27, 7638.57, 8324.07,
                    8589.7,  8727.01, 8914.71, 9255.67, 9321.01, 9561.95,
                    9904.28, 9983.38}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_03) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$2], dir0=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{37, 11, 33, 17, 2,  41, 47, 22, 1,  5,
                                    28, 35, 40, 21, 10, 15, 4,  23, 12, 32,
                                    48, 13, 50, 49, 16, 36, 26, 31, 14, 18,
                                    27, 46, 39, 38, 8,  25, 44, 3,  20, 29,
                                    6,  9,  34, 42, 19, 24, 30, 7,  43, 45}},
               {"GDF_INT32",
                Literals<GDF_INT32>{8,  23, 17, 2,  13, 10, 2,  3,  15, 3,
                                    8,  17, 3,  8,  5,  23, 4,  3,  13, 15,
                                    0,  3,  6,  10, 10, 21, 22, 23, 1,  6,
                                    3,  6,  2,  12, 17, 12, 16, 1,  22, 0,
                                    20, 8,  15, 5,  18, 13, 1,  18, 19, 9}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    -917.75, -272.6,  -78.56,  6.34,    121.65,  270.95,
                    274.58,  591.98,  711.56,  794.47,  1007.18, 1228.24,
                    1335.3,  1428.25, 2753.54, 2788.52, 2866.83, 3332.02,
                    3396.49, 3471.53, 3792.5,  3857.34, 4266.13, 4573.94,
                    4681.03, 4987.27, 5182.05, 5236.89, 5266.3,  5494.43,
                    5679.84, 5744.59, 6264.31, 6345.11, 6819.74, 7133.7,
                    7315.94, 7498.12, 7603.4,  7618.27, 7638.57, 8324.07,
                    8589.7,  8727.01, 8914.71, 9255.67, 9321.01, 9561.95,
                    9904.28, 9983.38}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_04) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_nationkey, c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$1], sort1=[$2], dir0=[ASC], dir1=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{48, 29, 14, 3,  30, 17, 47, 39, 22, 5,
                                    40, 23, 13, 27, 4,  10, 42, 50, 18, 46,
                                    37, 28, 21, 9,  45, 41, 49, 16, 38, 25,
                                    2,  12, 24, 1,  32, 34, 44, 33, 35, 8,
                                    19, 7,  43, 6,  36, 26, 20, 11, 15, 31}},
               {"GDF_INT32",
                Literals<GDF_INT32>{0,  0,  1,  1,  1,  2,  2,  2,  3,  3,
                                    3,  3,  3,  3,  4,  5,  5,  6,  6,  6,
                                    8,  8,  8,  8,  9,  10, 10, 10, 12, 12,
                                    13, 13, 13, 15, 15, 15, 16, 17, 17, 17,
                                    18, 18, 19, 20, 21, 22, 22, 23, 23, 23}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    3792.5,  7618.27, 5266.3,  7498.12, 9321.01, 6.34,
                    274.58,  6264.31, 591.98,  794.47,  1335.3,  3332.02,
                    3857.34, 5679.84, 2866.83, 2753.54, 8727.01, 4266.13,
                    5494.43, 5744.59, -917.75, 1007.18, 1428.25, 8324.07,
                    9983.38, 270.95,  4573.94, 4681.03, 6345.11, 7133.7,
                    121.65,  3396.49, 9255.67, 711.56,  3471.53, 8589.7,
                    7315.94, -78.56,  1228.24, 6819.74, 8914.71, 9561.95,
                    9904.28, 7638.57, 4987.27, 5182.05, 7603.4,  -272.6,
                    2788.52, 5236.89}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_05) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_nationkey, c_custkey",
      .logicalPlan =
          "LogicalSort(sort0=[$1], sort1=[$0], dir0=[ASC], dir1=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{29, 48, 3,  14, 30, 17, 39, 47, 5,  13,
                                    22, 23, 27, 40, 4,  10, 42, 18, 46, 50,
                                    9,  21, 28, 37, 45, 16, 41, 49, 25, 38,
                                    2,  12, 24, 1,  32, 34, 44, 8,  33, 35,
                                    7,  19, 43, 6,  36, 20, 26, 11, 15, 31}},
               {"GDF_INT32",
                Literals<GDF_INT32>{0,  0,  1,  1,  1,  2,  2,  2,  3,  3,
                                    3,  3,  3,  3,  4,  5,  5,  6,  6,  6,
                                    8,  8,  8,  8,  9,  10, 10, 10, 12, 12,
                                    13, 13, 13, 15, 15, 15, 16, 17, 17, 17,
                                    18, 18, 19, 20, 21, 22, 22, 23, 23, 23}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    7618.27, 3792.5,  7498.12, 5266.3,  9321.01, 6.34,
                    6264.31, 274.58,  794.47,  3857.34, 591.98,  3332.02,
                    5679.84, 1335.3,  2866.83, 2753.54, 8727.01, 5494.43,
                    5744.59, 4266.13, 8324.07, 1428.25, 1007.18, -917.75,
                    9983.38, 4681.03, 270.95,  4573.94, 7133.7,  6345.11,
                    121.65,  3396.49, 9255.67, 711.56,  3471.53, 8589.7,
                    7315.94, 6819.74, -78.56,  1228.24, 9561.95, 8914.71,
                    9904.28, 7638.57, 4987.27, 7603.4,  5182.05, -272.6,
                    2788.52, 5236.89}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
