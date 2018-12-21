#include <CalciteInterpreter.h>
#include <gdf/library/api.h>

#include <gtest/gtest.h>

class Item {
public:
    const std::string               logicalPlan;
    const std::int64_t              expectedOutputLength;
    const std::vector<std::int64_t> expectedCount;
};

class WithoutGroupByTest : public testing::TestWithParam<Item> {};

TEST_P(WithoutGroupByTest, RunQuery) {
    using namespace gdf::library;

    auto tableGroup = LiteralTableGroupBuilder{
      {"main.nation",
       {
         {"n_nationkey", Literals<GDF_INT64>{1, 3, 5, 1, 3, 7, 9}},
         {"n_regionkey", Literals<GDF_INT64>{0, 2, 4, 6, 8, 0, 2}},
       }}}.Build();

    auto inputTables = tableGroup.ToBlazingFrame();
    auto tableNames  = tableGroup.table_names();
    auto columnNames = tableGroup.column_names();

    std::uint8_t mask = 0b11111000;
    for (auto gdfColumnCpp : inputTables[0]) {
        gdf_column *gdfColumn = gdfColumnCpp.get_gdf_column();
        cudaMemcpy(gdfColumn->valid, &mask, 1, cudaMemcpyHostToDevice);
        gdfColumn->null_count = 5;
    }

    std::vector<gdf_column_cpp> outputs;

    gdf_error gdfError = evaluate_query(
      inputTables, tableNames, columnNames, GetParam().logicalPlan, outputs);

    EXPECT_EQ(GDF_SUCCESS, gdfError);
    EXPECT_EQ(GetParam().expectedOutputLength, outputs.size());

    auto outputTable =
      GdfColumnCppsTableBuilder{"outputTable", outputs}.Build();

    EXPECT_EQ(1, outputTable.size());
    EXPECT_EQ(GetParam().expectedCount.size(), outputTable[0].size());

    for (std::size_t i = 0; i < GetParam().expectedCount.size(); i++) {
        EXPECT_EQ(GetParam().expectedCount[i],
                  outputTable[0][i].get<GDF_INT64>())
          << "with index " << i;
    }
}

INSTANTIATE_TEST_CASE_P(
  QueriesWithCount,
  WithoutGroupByTest,
  testing::ValuesIn({
    Item{// select count(*) from main.nation
         "LogicalAggregate(group=[{}], EXPR$0=[COUNT()])\n"
         "  LogicalProject($f0=[0])\n"
         "    EnumerableTableScan(table=[[main, nation]])",
         1,
         {7}},
    Item{// select count(n_nationkey) from main.nation
         "LogicalAggregate(group=[{}], EXPR$0=[COUNT()])\n"
         "  LogicalProject(n_nationkey=[$0])\n"
         "    EnumerableTableScan(table=[[main, nation]])",
         1,
         {5}},
    Item{// select count(*) from main.nation group by n_nationkey
         "LogicalProject(EXPR$0=[$1])\n"
         "  LogicalAggregate(group=[{0}], EXPR$0=[COUNT()])\n"
         "    LogicalProject(n_nationkey=[$0])\n"
         "      EnumerableTableScan(table=[[main, nation]])",
         4,
         {2, 2, 2, 1}},
  }));
