
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
    std::vector<std::string> filePaths;
    std::vector<std::string> tableNames;
    std::vector<std::vector<std::string>> columnNames;
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
TEST_F(EvaluateQueryTest, TEST_01) {
  auto input = InputTestItem{
      .query =
          "select nation.n_nationkey, region.r_regionkey from main.nation "
          "inner join main.region on region.r_regionkey = nation.n_nationkey",
      .logicalPlan =
          "LogicalProject(n_nationkey=[$0], r_regionkey=[$4])\n  "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], r_regionkey=[$5], r_name=[$6], r_comment=[$7])\n    "
          "LogicalJoin(condition=[=($5, $4)], joinType=[inner])\n      "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], n_nationkey0=[CAST($0):BIGINT NOT NULL])\n        "
          "EnumerableTableScan(table=[[main, nation]])\n      "
          "EnumerableTableScan(table=[[main, region]])",
      .filePaths = {"/home/aocsa/blazingdb/tpch/1mb/nation.psv",
                    "/home/aocsa/blazingdb/tpch/1mb/region.psv"},
      .tableNames = {"nation", "region"},
      .columnNames = {{"n_nationkey", "n_name", "n_regionkey", "n_comment"},
                      {"r_regionkey", "r_name", "r_comment"}},
      .resultTable =
          LiteralTableBuilder{
              "ResultSet", {{"GDF_INT64", Literals<GDF_INT64>{0, 1, 2, 3, 4}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
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
