/*
 * ResultSetRepository.h
 *
 *  Created on: Sep 21, 2018
 *      Author: felipe
 */

#ifndef RESULTSETREPOSITORY_H_
#define RESULTSETREPOSITORY_H_

#include "DataFrame.h"
#include <map>
#include <vector>
#include <mutex>

typedef uint64_t query_token;
typedef uint64_t connection_id;
typedef void * response_descriptor; //this shoudl be substituted for something that can generate a response

//singleton class

class result_set_repository {
public:
	static result_set_repository & get_instance(){
		  static result_set_repository instance;
		  return instance;
	}
	virtual ~result_set_repository();
	result_set_repository();
	query_token register_query(connection_id connection);
	void update_token(query_token token,blazing_frame frame);
	void remove_all_connection_tokens(connection_id connection);
	void get_result(query_token token, response_descriptor response_to_write); //this last param is a dummy one that will take the infromation we need
																	// in order to write out this response back to the requester

	result_set_repository(result_set_repository const&)	= delete;
	void operator=(result_set_repository const&)		= delete;
private:
	std::map<query_token,std::tuple<bool,blazing_frame> > result_sets;
	std::map<connection_id,std::vector<query_token> > connection_result_sets;

	void add_token(query_token token, connection_id connection);
	std::map<query_token,response_descriptor> requested_responses;
	std::mutex repo_mutex;


};

#endif /* RESULTSETREPOSITORY_H_ */
