
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <StringUtil.h>
#include <gtest/gtest.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include <Utils.cuh>

#include "gdf/library/api.h"
using namespace gdf::library;

#include <gdf/cffi/functions.h>
#include <gdf/gdf.h>
#include <sys/stat.h>

#include "csv_utils.cuh"

using namespace gdf::library;

struct EvaluateQueryTest : public ::testing::Test {
  struct InputTestItem {
    std::string query;
    std::string logicalPlan;
    std::vector<std::string> filePaths;
    std::vector<std::string> tableNames;
    std::vector<std::vector<std::string>> columnNames;
    std::vector<std::vector<const char*>> columnTypes;
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
TEST_F(EvaluateQueryTest, TEST_00) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer where "
          "c_custkey < 15",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], c_nationkey=[$3], c_acctbal=[$5])\n  "
          "LogicalFilter(condition=[<($0, 15)])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64", Literals<GDF_INT64>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                 11, 12, 13, 14}},
               {"GDF_INT64", Literals<GDF_INT64>{15, 13, 1, 4, 3, 20, 18, 17, 8,
                                                 5, 23, 13, 3, 1}},
               {"GDF_INT64",
                Literals<GDF_FLOAT32>{711.56, 121.65, 7498.12, 2866.83, 794.47,
                                    7638.57, 9561.95, 6819.74, 8324.07, 2753.54,
                                    -272.6, 3396.49, 3857.34, 5266.3}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
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
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer where "
          "c_custkey < 150 and c_nationkey = 5",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], c_nationkey=[$3], c_acctbal=[$5])\n  "
          "LogicalFilter(condition=[AND(<($0, 150), =($3, 5))])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64", Literals<GDF_INT64>{10, 42, 85, 108, 123, 138}},
               {"GDF_INT64", Literals<GDF_INT64>{5, 5, 5, 5, 5, 5}},
               {"GDF_INT64", Literals<GDF_FLOAT32>{2753.54, 8727.01, 3386.64,
                                                 2259.38, 5897.83, 430.59}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
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
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "c_custkey < 0",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[<($0, 0)])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable = LiteralTableBuilder{"ResultSet", {}}.Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
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
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "c_custkey < 0 and c_nationkey >=30",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[AND(<($0, 0), >=($3, 30))])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable = LiteralTableBuilder{"ResultSet", {}}.Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
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
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "c_custkey < 0 or c_nationkey >= 24",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[OR(<($0, 0), >=($3, 24))])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT64", Literals<GDF_INT64>{117}},
                               {"GDF_INT64", Literals<GDF_INT64>{24}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
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
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "c_custkey < 0 and c_nationkey >= 3",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[AND(<($0, 0), >=($3, 3))])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable = LiteralTableBuilder{"ResultSet", {}}.Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_06) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "-c_nationkey + c_acctbal > 750.3",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[>(+(-($3), $5), 750.3)])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64",
                Literals<GDF_INT64>{
                    3,   4,   5,   6,   7,   8,   9,   10,  12,  13,  14,  15,
                    16,  18,  19,  20,  21,  23,  24,  25,  26,  27,  28,  29,
                    30,  31,  32,  34,  35,  36,  38,  39,  40,  42,  43,  44,
                    45,  46,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                    58,  59,  60,  61,  63,  65,  67,  68,  69,  70,  73,  74,
                    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,
                    87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  99,
                    100, 101, 102, 103, 105, 106, 107, 108, 110, 111, 112, 113,
                    114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 126, 127,
                    129, 130, 131, 133, 134, 135, 137, 139, 140, 141, 142, 143,
                    144, 145, 146, 147, 148, 149, 150}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    1,  4,  3,  20, 18, 17, 8,  5,  13, 3,  1,  23, 10, 6,  18,
                    22, 8,  3,  13, 12, 22, 3,  8,  0,  1,  23, 15, 15, 17, 21,
                    12, 2,  3,  5,  19, 16, 9,  6,  0,  10, 6,  12, 11, 15, 4,
                    10, 10, 21, 13, 1,  12, 17, 21, 23, 9,  12, 9,  22, 0,  4,
                    18, 0,  17, 9,  15, 0,  20, 18, 22, 11, 5,  0,  23, 16, 14,
                    16, 8,  2,  7,  9,  15, 8,  17, 15, 20, 2,  19, 9,  10, 1,
                    15, 5,  10, 22, 19, 12, 14, 8,  16, 24, 18, 7,  17, 3,  5,
                    18, 22, 21, 7,  9,  11, 17, 11, 19, 16, 9,  4,  1,  9,  16,
                    1,  13, 3,  18, 11, 19, 18}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_07) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey as nkey from main.customer where "
          "-c_nationkey + c_acctbal > 750",
      .logicalPlan =
          "LogicalProject(c_custkey=[$0], nkey=[$3])\n  "
          "LogicalFilter(condition=[>(+(-($3), $5), 750)])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .filePaths = {"../resources/tpch-generator/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64",
                Literals<GDF_INT64>{
                    3,   4,   5,   6,   7,   8,   9,   10,  12,  13,  14,  15,
                    16,  18,  19,  20,  21,  23,  24,  25,  26,  27,  28,  29,
                    30,  31,  32,  34,  35,  36,  38,  39,  40,  42,  43,  44,
                    45,  46,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                    58,  59,  60,  61,  63,  65,  67,  68,  69,  70,  73,  74,
                    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,
                    87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  99,
                    100, 101, 102, 103, 105, 106, 107, 108, 110, 111, 112, 113,
                    114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 126, 127,
                    129, 130, 131, 133, 134, 135, 137, 139, 140, 141, 142, 143,
                    144, 145, 146, 147, 148, 149, 150}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    1,  4,  3,  20, 18, 17, 8,  5,  13, 3,  1,  23, 10, 6,  18,
                    22, 8,  3,  13, 12, 22, 3,  8,  0,  1,  23, 15, 15, 17, 21,
                    12, 2,  3,  5,  19, 16, 9,  6,  0,  10, 6,  12, 11, 15, 4,
                    10, 10, 21, 13, 1,  12, 17, 21, 23, 9,  12, 9,  22, 0,  4,
                    18, 0,  17, 9,  15, 0,  20, 18, 22, 11, 5,  0,  23, 16, 14,
                    16, 8,  2,  7,  9,  15, 8,  17, 15, 20, 2,  19, 9,  10, 1,
                    15, 5,  10, 22, 19, 12, 14, 8,  16, 24, 18, 7,  17, 3,  5,
                    18, 22, 21, 7,  9,  11, 17, 11, 19, 16, 9,  4,  1,  9,  16,
                    1,  13, 3,  18, 11, 19, 18}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables =
      ToBlazingFrame(input.filePaths, input.columnNames, input.columnTypes);
  auto table_names = input.tableNames;
  auto column_names = input.columnNames;
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
