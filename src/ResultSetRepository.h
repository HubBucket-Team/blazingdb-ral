/*
 * ResultSetRepository.h
 *
 *  Created on: Sep 21, 2018
 *      Author: felipe
 */

#ifndef RESULTSETREPOSITORY_H_
#define RESULTSETREPOSITORY_H_

#include "DataFrame.h"
#include "Types.h"
#include <map>
#include <vector>
#include <mutex>
#include <condition_variable>

typedef void * response_descriptor; //this shoudl be substituted for something that can generate a response

//singleton class

class result_set_repository {
public:

	bool free_result(query_token_t token);
	virtual ~result_set_repository();
	result_set_repository();
	static result_set_repository & get_instance(){
		  static result_set_repository instance;
		  return instance;
	}

	query_token_t register_query(connection_id_t connection);
	void update_token(query_token_t token, blazing_frame frame);
	connection_id_t init_session();
	void remove_all_connection_tokens(connection_id_t connection);
	blazing_frame get_result(connection_id_t connection, query_token_t token);

	result_set_repository(result_set_repository const&)	= delete;
	void operator=(result_set_repository const&)		= delete;
private:
	std::map<query_token_t,std::tuple<bool, blazing_frame> > result_sets;
	std::map<connection_id_t,std::vector<query_token_t> > connection_result_sets;

	void add_token(query_token_t token, connection_id_t connection);
	std::map<query_token_t,response_descriptor> requested_responses;
	std::mutex repo_mutex;
	std::condition_variable cv;
};

#endif /* RESULTSETREPOSITORY_H_ */
