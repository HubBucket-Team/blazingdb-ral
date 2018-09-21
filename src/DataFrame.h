/*
 * DataFrame.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_


#include <gdf/gdf.h>
#include <GDFColumn.cuh>
#include <vector>
typedef struct blazing_frame{

public:
	gdf_column_cpp get_column(int column_index){
		size_t cur_count = 0;
		for(int i = 0; i < columns.size(); i++){
			if(column_index < cur_count + columns[i].size()){
				return columns[i][column_index - cur_count];
			}
			
			cur_count += columns[i].size();
		}
		//return nullptr; //error
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
	
	void consolidate_tables(){
		std::vector<gdf_column_cpp> new_tables;
		for(int table_index = 0; table_index < columns.size(); table_index++){
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
		return this->columns.size(); //ToDo check correctness
	}

	void clear(){
		this->columns.resize(0);
	}
private:
	std::vector<std::vector<gdf_column_cpp> > columns;
	//std::vector<gdf_column *> row_indeces; //per table row indexes used for materializing
} blazing_frame;



#endif /* DATAFRAME_H_ */
