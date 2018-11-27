/*
 * JoinProcessor.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef JOINPROCESSOR_H_
#define JOINPROCESSOR_H_

#include "gdf_wrapper/gdf_wrapper.cuh"
#include "DataFrame.h"

gdf_error evaluate_join(std::string condition,
		std::string join_type,
		blazing_frame data_frame,
		gdf_column * left_indices,
		gdf_column * right_indices
		);


#endif /* JOINPROCESSOR_H_ */
