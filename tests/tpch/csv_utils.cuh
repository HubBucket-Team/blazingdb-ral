#pragma once

#include <sys/stat.h>


#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include "gdf/library/types.h"

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


BlazingFrame ToBlazingFrame(std::vector<std::string> filePaths, std::vector<std::vector<std::string>> columnNames, std::vector<std::vector<const char*>> columnDTypes)
{
  BlazingFrame frame;
  for(size_t index = 0; index < filePaths.size(); index++) {
    auto file_path = filePaths[index];
    gdf_error error = GDF_SUCCESS;
  

    std::vector<const char*> columnNamesPointers;
    std::transform(columnNames[index].begin(), columnNames[index].end(), std::back_inserter(columnNamesPointers),
                   [](std::string &s)  { return s.c_str(); });



    if (checkFile(file_path.c_str())) {
    	csv_read_arg args{};
      args.filepath_or_buffer		= file_path.c_str();
      args.num_cols		=  columnNames[index].size();
      args.names			= columnNamesPointers.data();
      args.dtype			= columnDTypes[index].data();
      args.delimiter		= '|';
      args.lineterminator = '\n';
      args.decimal = '.';

      error = read_csv(&args);
      assert(error == GDF_SUCCESS);

      std::cout << "CSV output" << std::endl;
      std::cout << args.num_cols_out << std::endl;
      std::cout << args.num_rows_out << std::endl;

      frame.push_back(
        ToGdfColumnCpps(args.data, args.names, args.num_cols_out)
      );

    }
  }
  return frame;
}
