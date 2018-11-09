/*
 * CodeTimer.cpp
 *
 *  Created on: Mar 31, 2016
 *      Author: root
 */

#include "CodeTimer.h"

#include <iostream>

CodeTimer::CodeTimer() {
	start = std::chrono::high_resolution_clock::now();
}

CodeTimer::~CodeTimer() {
}

void CodeTimer::reset() {
	start = std::chrono::high_resolution_clock::now();
}

double CodeTimer::getDuration(){
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end-start).count();
}

void CodeTimer::display() {
	display("");
}

void CodeTimer::display(std::string msg) {
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double, std::milli>(end-start).count();
	std::cout<<"TIMING: "<<msg<<" | Duration: "<<duration<<"ms"<<std::endl;
}
