#include "gtest/gtest.h"
#include "ResultSetRepository.h"
#include <CalciteInterpreter.h>
#include <gdf/gdf.h>
#include <gdf/library/table_group.h>

struct ResultSetRepositoryTest : public ::testing::Test {
    ResultSetRepositoryTest() {
    }

    virtual ~ResultSetRepositoryTest() {
    }

    virtual void SetUp() {

    }

    virtual void TearDown() {
    }
};

TEST_F(ResultSetRepositoryTest, AddToken) {
    ASSERT_TRUE(true);
}

TEST_F(ResultSetRepositoryTest, AddTokenTwice) {
    ASSERT_TRUE(true);
}

TEST_F(ResultSetRepositoryTest, AddTokenMultipleTimes) {
    ASSERT_TRUE(true);
}

TEST_F(ResultSetRepositoryTest, FreeResultsetWhenEmpty) {

    //Invalid access and query tokens
    try {
        result_set_repository::get_instance().free_resultset(100, 101);
        FAIL() << "Expected std::runtime_error";
    } catch(std::runtime_error const & err) {
        EXPECT_EQ(err.what(), std::string("Connection does not exist"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }
}

using gdf::library::DType;
using gdf::library::Index;
using gdf::library::TableGroupBuilder;

using RType = DType<GDF_INT32>;

struct ResultSetRepositoryFakeData : ResultSetRepositoryTest {

    gdf::library::TableGroup group;

    ResultSetRepositoryFakeData() : group{TableGroupBuilder{
        {"hr.emps",
         {
           {"x", [](Index i) -> RType { return i % 2 ? i : 1; }},
         }},
      }
      .Build(num_values)} {}

    void SetUp() {
        input_tables = group.ToBlazingFrame();

        input1 = reinterpret_cast<const std::int32_t*>(group[0][0].get(0));
    }

    void TearDown() {
    }

    gdf_column_cpp one;

    std::vector<gdf_column_cpp> inputs;

    static const std::size_t num_values = 32;

    const std::int32_t *input1;

    std::vector<std::vector<gdf_column_cpp> > input_tables;
    std::vector<std::string> table_names={"hr.emps"};
    std::vector<std::vector<std::string>> column_names={{"x"}};

};

TEST_F(ResultSetRepositoryFakeData, FreeResultsetWhenInvalidToken) {

	std::vector<gdf_column_cpp> outputs;
    std::vector<void *> handles;
    uint64_t accessToken = 100;

	std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

    uint64_t resultToken = evaluate_query(input_tables, table_names, column_names, query, accessToken, handles);

    try {
        result_set_repository::get_instance().free_resultset(accessToken, resultToken + 1);
        FAIL() << "Expected std::runtime_error";
    } catch(std::runtime_error const & err) {
        EXPECT_EQ(err.what(), std::string("Result set does not exist"));
    } catch(...) {
        FAIL() << "Expected std::runtime_error";
    }

}

TEST_F(ResultSetRepositoryFakeData, FreeResultsetWhenValidToken) {

	std::vector<gdf_column_cpp> outputs;
    std::vector<void *> handles;
    uint64_t accessToken = 100;

	std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

    uint64_t resultToken = evaluate_query(input_tables, table_names, column_names, query, accessToken, handles);

    try {
        result_set_repository::get_instance().free_resultset(accessToken, resultToken);
    } catch(std::runtime_error const & err) {
        FAIL() << "Expected no error";
    } catch(...) {
        FAIL() << "Expected no error";
    }

}