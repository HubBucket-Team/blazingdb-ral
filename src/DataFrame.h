/*
 * DataFrame.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_


#include "Utils.cuh"
#include "gdf_wrapper/gdf_wrapper.cuh"
#include <GDFColumn.cuh>
#include <vector>
typedef struct blazing_frame{

public:
	// @todo: constructor copia, operator = 
	blazing_frame() 
		: 	columns {}
	{
	}

	blazing_frame(const blazing_frame& other) 
		: 	columns {other.columns}
	{
	}

	blazing_frame (blazing_frame&& other) 
		: columns { std::move (other.columns) }
	{

	}

	blazing_frame& operator = (const blazing_frame& other) {
		this->columns = other.columns;
		return *this;
	}

	blazing_frame& operator = (blazing_frame&& other) {
		this->columns = std::move(other.columns);
		return *this;
	}


	gdf_column_cpp & get_column(int column_index){

		if (column_index < 0) {
			std::cout << "index get off range." << std::endl;
 			return columns[0][0];
		}

		int cur_count = 0;
		for (std::size_t i = 0; i < columns.size(); i++){
			if ( column_index < cur_count + static_cast<int>(columns[i].size()) ){ // cast to int columns
				return columns[i][column_index - cur_count];
			}
			cur_count += columns[i].size();
		}

		std::cout << "index get off range." << std::endl;
		return columns[columns.size() - 1][cur_count - 1];
	}

	std::vector< std::vector<gdf_column_cpp> > get_columns(){
		return columns;
	}
	void add_table(std::vector<gdf_column_cpp> columns_to_add){
		columns.push_back(columns_to_add);
		if(columns_to_add.size() > 0){
			//fill row_indeces with 0 to n
		}
	}



	void set_column(size_t column_index, gdf_column_cpp column){
		size_t cur_count = 0;
		for(std::size_t i = 0; i < columns.size(); i++){
			if(column_index < cur_count + columns[i].size()){
				columns[i][column_index - cur_count] = column;
			}

			cur_count += columns[i].size();
		}
	}

	void consolidate_tables(){
		std::vector<gdf_column_cpp> new_tables;
		for(std::size_t table_index = 0; table_index < columns.size(); table_index++){
			new_tables.insert(new_tables.end(),columns[table_index].begin(),
					columns[table_index].end());
		}
		this->columns.resize(1);
		this->columns[0] = new_tables;
	}

	void add_column(gdf_column_cpp column_to_add, int table_index=0){
		columns[table_index].push_back(column_to_add);
	}

	void remove_table(size_t table_index){
		columns.erase(columns.begin() + table_index);
	}

	void swap_table(std::vector<gdf_column_cpp> columns_to_add, size_t index){
		columns[index] = columns_to_add;
	}

	size_t get_size_column(int table_index=0) {
		return columns[table_index].size();
	}

	size_t get_width(){
		return get_size_columns();
	}

	size_t get_size_columns(){
		size_t size_columns = 0;
		for(std::size_t i = 0; i < columns.size(); i++){
			size_columns += columns[i].size();
		}

		return size_columns;
	}

	void clear(){
		this->columns.resize(0);
	}

	void print(std::string title)
	{
		std::cout<<"---> "<<title<<std::endl;
		for(std::size_t table_index = 0; table_index < columns.size(); table_index++)
		{
			std::cout<<"Table: "<<table_index<<"\n";
			for(std::size_t column_index = 0; column_index < columns[table_index].size(); column_index++)
				print_column<int32_t>(columns[table_index][column_index].get_gdf_column());
		}
	}

	// This function goes over all columns in the data frame and makes sure that no two columns are actually pointing to the same data, and if so, clones the data so that they are all pointing to unique data pointers
	void deduplicate(){
		std::map<void*,std::pair<int, int>>  dataPtrs; // keys are the pointers, value is the table and column index it came from
		for(std::size_t table_index = 0; table_index < columns.size(); table_index++) {
			for(std::size_t column_index = 0; column_index < columns[table_index].size(); column_index++) {
				auto it = dataPtrs.find(columns[table_index][column_index].get_gdf_column()->data);
				if (it != dataPtrs.end() ){ // found a duplicate
					columns[table_index][column_index] = columns[table_index][column_index].clone();
				}
				dataPtrs[columns[table_index][column_index].get_gdf_column()->data] = std::make_pair(table_index,column_index);
			}
		}
	}

private:
	std::vector<std::vector<gdf_column_cpp> > columns;
	//std::vector<gdf_column *> row_indeces; //per table row indexes used for materializing
} blazing_frame;



#endif /* DATAFRAME_H_ */
