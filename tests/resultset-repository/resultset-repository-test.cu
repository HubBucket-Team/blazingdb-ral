#include <type_traits>

#include <gtest/gtest.h>

#include <ResultSetRepository.h>
#include <GDFColumn.cuh>

#include "../utils/gdf/library/table_group.h"
#include <DataFrame.h>
#include <vector>

template <class T>
class ResultSetRepositoryTest : public ::testing::Test {
  virtual void SetUp() {
    auto repo = result_set_repository::get_instance();
    connection = repo.init_session();
    token = repo.register_query(connection);

  }

  virtual void TearDown(){
    auto repo = result_set_repository::get_instance();
    repo.remove_all_connection_tokens(connection);
  }
protected:
  connection_id_t connection;
  query_token_t token;
};

TEST_F(ResultSetRepositoryTest, basic_resulset_test) {

  {

    gdf_column_cpp column;
    column.create_gdf_column(GDF_INT32,100,nullptr,4);
    blazing_frame frame;
    std::vector<gdf_column_cpp> columns;
    columns.push_back(column.clone());
    frame.add_table(columns);
    result_set_repository repo = result_set_repository::get_instance();
    void update_token(query_token_t token, blazing_frame frame, double duration, std::string errorMsg = "");
    repo.update_token(token, frame , .01);

    result_set_t result = repo.get_result(connection,token);
    EXPECT_TRUE(result.is_ready);
    repo.free_result(connection,token);

    try {
        result = repo.get_result(connection,token);
        EXPECT_TRUE(false);
       }
       catch(std::runtime_error const & err) {
           EXPECT_EQ(err.what(),std::string("Result set does not exist"));
       }

  }
}


