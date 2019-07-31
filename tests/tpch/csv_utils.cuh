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
  column_cpp.set_name(name);
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
    case GDF_STRING:
    {
      return 0;
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

std::vector<gdf_column_cpp> ToGdfColumnCpps(const cudf::table& table)   {
  std::vector<gdf_column_cpp> gdfColumnsCpps;
  for(size_t i = 0; i < table.num_columns(); i++ ){
    const gdf_column	*column = table.get_column(i);
    if(column->dtype == GDF_STRING){
      size_t type_size = dtype_size(column->dtype);
      gdf_column_cpp column_cpp;
      NVStrings * string = static_cast<NVStrings *>(column->data);
      NVCategory * category = NVCategory::create_from_strings(*string);
      size_t num_values = category->size();
      column_cpp.create_gdf_column(category,num_values,std::string(column->col_name));
      gdfColumnsCpps.push_back(column_cpp);
    }else{
      size_t type_size = dtype_size(column->dtype);
      gdfColumnsCpps.push_back(ToGdfColumnCpp(column->col_name, column->dtype, column->size, column->data, type_size));
    }

  }
  return gdfColumnsCpps;
}

// BlazingFrame ToBlazingFrame(gdf_column	**data, const char	**names, size_t ncols)   {
//   BlazingFrame frame;
//   frame.push_back(
//     ToGdfColumnCpps(data, names, ncols)
//   );
//   return frame;
// }


BlazingFrame ToBlazingFrame(std::vector<std::string> filePaths, std::vector<std::vector<std::string>> columnNames, std::vector<std::vector<const char*>> columnDTypes)
{
  BlazingFrame frame;
  for(size_t index = 0; index < filePaths.size(); index++) {
    auto file_path = filePaths[index];

    std::vector<std::string> dtypes;
    for (auto* c_str : columnDTypes[index]){
      dtypes.push_back(c_str);
    }
    

    if (checkFile(file_path.c_str())) {
    	cudf::io::csv::reader_options args{};
      args.input_data_form = FILE_PATH;
      args.filepath_or_buffer		= file_path.c_str();
      args.names			= columnNames[index];
      args.dtype			= dtypes;
      
      args.delimiter = '|';
      args.lineterminator = '\n';
      args.quotechar = '"';
      args.quoting = cudf::io::csv::quote_style::QUOTE_MINIMAL;
      args.doublequote = false;
      args.delim_whitespace = false;
      args.skipinitialspace = false;
      args.dayfirst = false;
      args.mangle_dupe_cols = true;
      args.compression = "none";
      args.decimal = '.';
      // args.thousands
      args.skip_blank_lines = true;
      // args.comment
      args.keep_default_na = true;
      args.na_filter = false;
      // args.prefix
      args.header = -1;	

	    cudf::io::csv::reader csv_reader(args);
      cudf::table outTable = csv_reader.read();

      frame.push_back(
        ToGdfColumnCpps(outTable)
      );

    }
  }
  return frame;
}
