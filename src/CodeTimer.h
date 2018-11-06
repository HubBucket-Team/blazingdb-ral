/*
 * CodeTimer.h
 *
 *  Created on: Mar 31, 2016
 *      Author: root
 */

#ifndef CODETIMER_H_
#define CODETIMER_H_

#include <string>
#include <chrono>

class CodeTimer {
public:
	CodeTimer();
	virtual ~CodeTimer();
	void reset();
	void display();
	double getDuration();
	void display(std::string msg);

private:
	std::chrono::high_resolution_clock::time_point start;
};
#endif /* CODETIMER_H_ */
