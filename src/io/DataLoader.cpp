
#include "DataLoader.h"
#include "Traits/RuntimeTraits.h"
#include <blazingdb/io/Library/Logging/Logger.h>

#include <CodeTimer.h>

namespace ral {
namespace io {

//TODO: remove this hack i put this in here because this wasn't refactored before
//including the file that defines it causes conflicts becuase we are using string util in both libraries
//we need to fix that probably my moving stringutil out of both, or making it a header only library
size_t get_width_dtype(gdf_dtype type){
  return ral::traits::get_dtype_size_in_bytes(type);
}

data_loader::data_loader(std::shared_ptr<data_parser> _parser, std::shared_ptr<data_provider> _data_provider): provider(_data_provider), parser(_parser)  {

}


data_loader::~data_loader() {

}



void data_loader::load_data(std::vector<gdf_column_cpp> & columns, const std::vector<size_t>  & column_indices, const Schema & schema){
	static CodeTimer timer;
	timer.reset();

	std::vector<std::vector<gdf_column_cpp> > columns_per_file; //stores all of the columns parsed from each file
	//iterates through files and parses them into columns
	while(this->provider->has_next()){
		std::vector<gdf_column_cpp> converted_data;
		//a file handle that we can use in case errors occur to tell the user which file had parsing issues
		std::string user_readable_file_handle = this->provider->get_current_user_readable_file_handle();
		std::shared_ptr<arrow::io::RandomAccessFile> file = this->provider->get_next();

		if(file != nullptr){
			parser->parse(file, user_readable_file_handle, converted_data,schema,column_indices);

			columns_per_file.push_back(converted_data);
		}else{
			std::cout<<"Was unable to open "<<user_readable_file_handle<<std::endl;
		}
	}

	//checking if any errors occurred
	std::vector<std::string> provider_errors = this->provider->get_errors();
	if(provider_errors.size() != 0){
		for(size_t error_index = 0; error_index < provider_errors.size(); error_index++){
			std::cout<<provider_errors[error_index]<<std::endl;
		}
	}

	this->provider->reset();

	size_t num_columns, num_files = columns_per_file.size();

	if(num_files>0)
		num_columns = columns_per_file[0].size();

	if(num_files == 0 || num_columns == 0){ 	//we got no data

		parser->parse(nullptr, "", columns,schema,column_indices);
		return ;
	}

	//be replacing no longer needed gdf_column_cpp with this dummy column we can
	//make columns go out of scope while still preserving the size of the vector
	gdf_column_cpp dummy_column;

	if(num_files == 1){ 	//we have only one file so we can just return the columns we parsed from that file
		columns = columns_per_file[0];

	}else{ 	//we have more than one file so we need to concatenate
		size_t total_row_count = 0;
		for(size_t file_index = 0; file_index < num_files; file_index++){
			total_row_count += columns_per_file[file_index][0].size();
		}

		columns.resize(num_columns);
		for(size_t column_index = 0; column_index < num_columns; column_index++){
			//collect the columns into an array for the concat function
			gdf_column * columns_to_concat[num_files];
			for(size_t file_index = 0; file_index < num_files; file_index++){
				columns_to_concat[file_index] = columns_per_file[file_index][column_index].get_gdf_column();
			}

			//allocate space for the output
			gdf_column_cpp& col = columns_per_file[0][column_index];
			gdf_column_cpp concatColumn;
			if (std::any_of(columns_to_concat, columns_to_concat + num_files, [](auto* c){ return c->valid != nullptr; }))
				concatColumn.create_gdf_column(col.dtype(), total_row_count, nullptr, ral::io::get_width_dtype(col.dtype()), col.name());
			else
				concatColumn.create_gdf_column(col.dtype(), total_row_count, nullptr, nullptr, ral::io::get_width_dtype(col.dtype()), col.name());
			columns[column_index] = concatColumn;

			CUDF_CALL(gdf_column_concat(concatColumn.get_gdf_column(),	columns_to_concat, num_files));

			//make the column that was parsed from the file go out of scope to get freed
			for(size_t file_index = 0; file_index < num_files; file_index++){
				columns_per_file[file_index][column_index] = dummy_column;
			}
		}
	}
	Library::Logging::Logger().logInfo("-> data_loader::load_data  " + std::to_string(timer.getDuration()) + " ms");
}

void data_loader::get_schema(Schema & schema){
	this->parser->parse_schema(this->provider->get_all(),schema);
}

} /* namespace io */
} /* namespace ral */
