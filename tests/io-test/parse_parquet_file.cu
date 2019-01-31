#include <gtest/gtest.h>



#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <gdf_wrapper/gdf_wrapper.cuh>
#include "io/data_parser/ParquetParser.h"
#include "io/data_provider/UriDataProvider.h"
#include "io/data_parser/DataParser.h"
#include "io/data_provider/DataProvider.h"

#include <DataFrame.h>
#include <fstream>


#include <GDFColumn.cuh>

#include <cuio/parquet/api.h>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <boost/filesystem.hpp>

#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>
 

#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Library/Logging/CoutOutput.h>
#include "blazingdb/io/Library/Logging/ServiceLogging.h"

class ParquetReaderAPITest : public testing::Test {
protected:
    ParquetReaderAPITest()
        : filename("/home/rommel/blazingsql_workspace/DataSet50Mb/lineitem_0_0.parquet") {}

    const std::string filename;

    gdf_column *columns        = nullptr;
    std::size_t columns_length = 0;
};
 

TEST_F(ParquetReaderAPITest, ByIdsInFromInterface) {

    std::shared_ptr<::arrow::io::ReadableFile> file;
    const ::parquet::ReaderProperties          properties =
      ::parquet::default_reader_properties();
    ::arrow::io::ReadableFile::Open(filename, properties.memory_pool(), &file);

    
    std::vector<Uri> uris = {Uri{this->filename}};
    // this works
    //std::vector<bool> include_column = {false, false, false, true, false, false, false, false, false, false, true, false, false, false, false, false}; //lineitem
    // this will crash
    std::vector<bool> include_column = {true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false}; //lineitem
    std::unique_ptr<ral::io::data_provider> provider = std::make_unique<ral::io::uri_data_provider>(uris);
    std::unique_ptr<ral::io::data_parser> parser = std::make_unique<ral::io::parquet_parser>();


    EXPECT_TRUE(provider->has_next());
    std::vector<gdf_column_cpp> gdf_columns_cpp;
    parser->parse(provider->get_next(), gdf_columns_cpp, include_column);

    for(size_t column_index = 0; column_index < gdf_columns_cpp.size(); column_index++){
        std::cout << "col_name: " << gdf_columns_cpp[column_index].get_gdf_column()->col_name << std::endl;
        //print_gdf_column(gdf_columns_cpp[column_index].get_gdf_column());
    }
}

