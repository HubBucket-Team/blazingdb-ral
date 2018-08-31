/*
 * DataFrame.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_


#include <gdf/gdf.h>

typedef struct blazing_frame{

public:
	gdf_column * get_column(int column_index){
		size_t cur_count = 0;
		for(int i = 0; i < columns.size(); i++){
			if(column_index < cur_count + columns[i].size()){
				return columns[i][column_index - cur_count];
			}
			
			cur_count += columns[i].size();
		}
		return nullptr; //error
	}

	void add_table(std::vector<gdf_column * > columns_to_add){
		columns.push_back(columns_to_add);
		if(columns_to_add.size() > 0){
			//fill row_indeces with 0 to n
		}
	}
	
	void add_column(gdf_column * column_to_add, int table_index=0){
		columns[table_index].push_back(column_to_add);
	}

	void remove_table(size_t table_index){
		columns.erase(columns.begin() + table_index);
	}

	void swap_table(std::vector<gdf_column * > columns_to_add, size_t index){
		columns[index] = columns_to_add;
	}
	
	size_t get_size_column(int table_index=0) {
		return columns[table_index].size();
	}
	
	size_t get_width(){
		size_t width = 0;

	}

	void clear(){
		this->columns.resize(0);
	}
private:
	std::vector<std::vector<gdf_column *> > columns;
	//std::vector<gdf_column *> row_indeces; //per table row indexes used for materializing
} blazing_frame;



#endif /* DATAFRAME_H_ */
