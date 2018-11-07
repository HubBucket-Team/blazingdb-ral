
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
      .filePaths = {"/home/aocsa/blazingdb/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32", Literals<GDF_INT32>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                 11, 12, 13, 14}},
               {"GDF_INT32", Literals<GDF_INT32>{15, 13, 1, 4, 3, 20, 18, 17, 8,
                                                 5, 23, 13, 3, 1}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{711.56, 121.65, 7498.12, 2866.83, 794.47,
                                      7638.57, 9561.95, 6819.74, 8324.07,
                                      2753.54, -272.6, 3396.49, 3857.34,
                                      5266.3}}}}
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
      .filePaths = {"/home/aocsa/blazingdb/tpch/1mb/customer.psv"},
      .tableNames = {"main.customer"},
      .columnNames = {{"c_custkey", "c_name", "c_address", "c_nationkey",
                       "c_phone", "c_acctbal", "c_mktsegment", "c_comment"}},
      .columnTypes = {{"int32", "int64", "int64", "int32", "int64", "float32",
                       "int64", "int64"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32", Literals<GDF_INT32>{10, 42, 85, 108, 123, 138}},
               {"GDF_INT32", Literals<GDF_INT32>{5, 5, 5, 5, 5, 5}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{2753.54, 8727.01, 3386.64, 2259.38,
                                      5897.83, 430.59}}}}
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
