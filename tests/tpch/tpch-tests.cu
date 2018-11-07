#include <gtest/gtest.h>
#include <sys/stat.h>

#include "gdf/library/table_group.h"
#include "gdf/library/types.h"

#include <gdf/cffi/functions.h>
#include <gdf/gdf.h>
using namespace gdf::library;

bool checkFile(const char *fpath) {
	struct stat     st;

	if (stat(fpath, &st)) {
		return 0;
	}
	return 1;
}


gdf_column_cpp ToGdfColumnCpp(const std::string &name,
                              const gdf_dtype    dtype,
                              const std::size_t  length,
                              const void *       data,
                              const std::size_t  size) {
  gdf_column_cpp column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  column_cpp.delete_set_name(name);
  return column_cpp;
}


int dtype_size(gdf_dtype col_type) {
  switch( col_type )
    {
    case GDF_INT8:
      {
        using ColType = int8_t;

        return sizeof(ColType);
      }
    case GDF_INT16:
      {
        using ColType = int16_t;

        return sizeof(ColType);
      }
    case GDF_INT32:
      {
        using ColType = int32_t;

        return sizeof(ColType);
      }
    case GDF_INT64:
      {
        using ColType = int64_t;

        return sizeof(ColType);
      }
    case GDF_FLOAT32:
      {
        using ColType = float;

        return sizeof(ColType);
      }
    case GDF_FLOAT64:
      {
        using ColType = double;

        return sizeof(ColType);
      }

    default:
      assert( false );//type not handled
    }
    return 0;
}

std::vector<gdf_column_cpp> ToGdfColumnCpps(gdf_column	**data, const char	**names, size_t ncols)   {
  std::vector<gdf_column_cpp> gdfColumnsCpps;
  for(size_t i = 0; i < ncols; i++ ){
    gdf_column	*column = data[i];
    size_t type_size = dtype_size(column->dtype);
    gdfColumnsCpps.push_back(ToGdfColumnCpp(names[i], column->dtype, column->size, column->data, type_size));
  }
  return gdfColumnsCpps;
}

BlazingFrame ToBlazingFrame(gdf_column	**data, const char	**names, size_t ncols)   {
  BlazingFrame frame;
  frame.push_back(
    ToGdfColumnCpps(data, names, ncols)
  );
  return frame;
}



TEST(UtilsTest, CSVReaderGdf)
{
  gdf_error error = GDF_SUCCESS;
  csv_read_arg args;
  args.num_cols = 8;
  args.names = new const char* [8] {
    "c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"
  };
  args.dtype = new const char* [8] {"int32", "int64", "int64", "int32", "int64", "float32", "int64", "int64"};
  args.file_path = (char*)("/home/aocsa/blazingdb/tpch/1mb/customer.psv");

  if (checkFile(args.file_path)) {

    args.delimiter = '|';
    args.lineterminator = '\n';
    args.delim_whitespace = 0;
    args.skipinitialspace = 0;
    args.skiprows = 0;
    args.skipfooter = 0;
    args.dayfirst = 0;

    error = read_csv(&args);

    std::cout << args.num_cols_out << std::endl;
    std::cout << args.num_rows_out << std::endl;
    auto bframe = ToBlazingFrame(args.data, args.names, args.num_cols_out);

  }

  EXPECT_TRUE(error == GDF_SUCCESS);
}