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
		}
	}

	void add_table(std::vector<gdf_column * > columns_to_add){
		columns.push_back(columns_to_add);
		if(columns_to_add.size() > 0){
			//fill row_indeces with 0 to n
		}
	}

	void remove_table(size_t table_index){

		columns.erase(columns.begin() + table_index);
	}

	void swap_table(std::vector<gdf_column * > columns_to_add, size_t index){
		columns[index] = columns_to_add;
	}
private:
	std::vector<std::vector<gdf_column *> > columns;
	std::vector<gdf_column *> row_indeces; //per table row indexes used for materializing
} blazing_frame;



#endif /* DATAFRAME_H_ */
