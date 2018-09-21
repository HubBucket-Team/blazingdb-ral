/*
 * ResultSetRepository.cpp
 *
 *  Created on: Sep 21, 2018
 *      Author: felipe
 */

#include "ResultSetRepository.h"
#include <random>

result_set_repository::result_set_repository() {
	// nothing really has to be instantiated

}

result_set_repository::~result_set_repository() {
	//nothing needs to be destroyed
}

void result_set_repository::add_token(query_token token, connection_id connection){
	blazing_frame temp;
	this->result_sets[token] = std::make_tuple(false,temp);

	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		std::vector<query_token> empty_tokens;
		this->connection_result_sets[connection] = empty_tokens;
	}

	this->connection_result_sets[connection].push_back(token);
}

query_token result_set_repository::register_query(connection_id connection){
	  std::random_device rd;
	  std::mt19937 gen(rd());
	  std::uniform_int_distribution<query_token> dis(
			  std::numeric_limits<query_token>::min(),
			  std::numeric_limits<query_token>::max());

	  query_token token = dis(gen);
	  this->add_token(token,connection);
	  return token;
}

void write_response(blazing_frame frame,response_descriptor response_to_write){
	//TODO: use flatbuffers here to convert the frame to the response message
}
void result_set_repository::update_token(query_token token, blazing_frame frame){
	if(this->result_sets.find(token) == this->result_sets.end()){
		//TODO: throw an error here
	}
	std::lock_guard<std::mutex> guard(this->repo_mutex);
	this->result_sets[token] = std::make_tuple(true,frame);
	if(this->requested_responses.find(token) != this->requested_responses.end()){
		write_response(std::get<1>(this->result_sets[token]),this->requested_responses[token]);
	}
}

void result_set_repository::remove_all_connection_tokens(connection_id connection){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		//TODO: throw some error we are clsoing a connectiont hat did not exist
	}

	for(query_token token : this->connection_result_sets[connection]){
		this->result_sets.erase(token);
	}
	this->connection_result_sets.erase(connection);
}

void result_set_repository::get_result(query_token token, response_descriptor response_to_write){
	if(this->result_sets.find(token) == this->result_sets.end()){
		//TODO: throw some error you wanta result set that doesnt exist
	}
	{
		//scope the lockguard here
		std::lock_guard<std::mutex> guard(this->repo_mutex);
		if(std::get<0>(this->result_sets[token])){
			write_response(std::get<1>(this->result_sets[token]),response_to_write);
		}else{
			this->requested_responses[token] = response_to_write;
		}
	}

}

