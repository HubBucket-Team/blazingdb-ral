/*
 * StringUtil.cpp
 *
 *  Created on: Aug 7, 2014
 *      Author: felipe
 */

#include "StringUtil.h"
#include <sstream>
#include <algorithm>

StringUtil::StringUtil() {
	// TODO Auto-generated constructor stub
}

StringUtil::~StringUtil() {
	// TODO Auto-generated destructor stub
}

std::vector<std::string> StringUtil::split(std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

std::vector<std::string> & StringUtil::split(std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
    	if (!(item.size() == 1 && item[0] == delim)){
    		elems.push_back(item);
    	}
    }
    if(s[s.size()-1] == delim){
    	elems.push_back("");
    }
    return elems;
}

// trim from end
std::string & StringUtil::rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}