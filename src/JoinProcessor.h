/*
 * JoinProcessor.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef JOINPROCESSOR_H_
#define JOINPROCESSOR_H_


#include <gdf/gdf.h>
#include "DataFrame.h"

gdf_error process_join(std::string condition,
		std::string join_type,
		blazing_frame data_frame,
		gdf_column * left_indeces,
		gdf_column * right_indeces
		);


#endif /* JOINPROCESSOR_H_ */
