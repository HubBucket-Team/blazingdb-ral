
#include "DataLoader.h"

namespace ral {
namespace io {

//TODO: remove this hack i put this in here because this wasn't refactored before
//including the file that defines it causes conflicts becuase we are using string util in both libraries
//we need to fix that probably my moving stringutil out of both, or making it a header only library
size_t get_width_dtype(gdf_dtype type){
	if(type == GDF_INT8){
		return 1;
	}else if(type == GDF_INT16){
		return 2;
	}else if(type == GDF_INT32){
		return 4;
	}else if(type == GDF_INT64){
		return 8;
	//}
	//TODO felipe percy noboa see upgrade to uints
//	else if(type == GDF_UINT8){
//		return 1;
//	}else if(type == GDF_UINT16){
//		return 2;
//	}else if(type == GDF_UINT32){
//		return 4;
//	}else if(type == GDF_UINT64){
//		return 8;
	}else if(type == GDF_FLOAT32)
	{
		return 4;
	}else if(type == GDF_FLOAT64){
		return 8;
	}else if(type == GDF_DATE32){
		return 4;
	}else if(type == GDF_DATE64){
		return 8;
	}else if(type == GDF_TIMESTAMP){
		return 8;
	}else if(type == GDF_CATEGORY){
		return 0;
	}else if(type == GDF_STRING){
		return 0;
	}else{
		return 0;
	}
}

data_loader::data_loader(std::shared_ptr<data_parser> _parser, std::shared_ptr<data_provider> _data_provider): provider(_data_provider), parser(_parser)  {

}


data_loader::~data_loader() {

}



void data_loader::load_data(std::vector<gdf_column_cpp> & columns, const std::vector<size_t>  & column_indices, const Schema & schema){

	std::vector<std::vector<gdf_column_cpp> > columns_per_file; //stores all of the columns parsed from each file
	//iterates through files and parses them into columns
	size_t file_index = 0;
	while(this->provider->has_next()){
		std::vector<gdf_column_cpp> converted_data;
		//a file handle that we can use in case errors occur to tell the user which file had parsing issues
		std::string user_readable_file_handle = this->provider->get_current_user_readable_file_handle();
		std::shared_ptr<arrow::io::RandomAccessFile> file = this->provider->get_next();

		if(file != nullptr){
			parser->parse(file,converted_data,schema,column_indices,file_index);

			columns_per_file.push_back(converted_data);
		}else{
			std::cout<<"Was unable to open "<<user_readable_file_handle<<std::endl;
		}
		file_index++;
	}

	//checking if any errors occurred
	std::vector<std::string> provider_errors = this->provider->get_errors();
	if(provider_errors.size() != 0){
		for(size_t error_index = 0; error_index < provider_errors.size(); error_index++){
			std::cout<<provider_errors[error_index]<<std::endl;
		}
	}

	size_t num_columns, num_files = columns_per_file.size();

	if(num_files>0)
		num_columns = columns_per_file[0].size();

	if(num_files == 0 || num_columns == 0){ 	//we got no data

		parser->parse(nullptr,columns,schema,column_indices,file_index);
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
			//allocate space for the output
			gdf_column_cpp column;
			column.create_gdf_column(columns_per_file[0][column_index].dtype(),
					total_row_count,
					nullptr,
					ral::io::get_width_dtype(columns_per_file[0][column_index].dtype()),
					columns_per_file[0][column_index].name());
			columns[column_index] = column;

			//collect the columns into an array for the concat function
			gdf_column * columns_to_concat[num_files];
			for(size_t file_index = 0; file_index < num_files; file_index++){
				columns_to_concat[file_index] = columns_per_file[file_index][column_index].get_gdf_column();
			}

			gdf_error err = gdf_column_concat(column.get_gdf_column(),
					columns_to_concat,
					num_files);

			//make the column that was parsed from the file go out of scope to get freed
			for(size_t file_index = 0; file_index < num_files; file_index++){
				columns_per_file[file_index][column_index] = dummy_column;
			}

			if(err != GDF_SUCCESS){
				columns.resize(0);
				//TODO: do something better than this, we should proably be throwing errors and handling them up the stack
				std::cout<<"Error when trying to concatenate columns"<<std::endl;
			}
		}

	}

}

void data_loader::get_schema(Schema & schema){
	this->parser->parse_schema(this->provider->get_all(),schema);
}

} /* namespace io */
} /* namespace ral */
