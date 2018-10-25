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

std::string StringUtil::replace(std::string containingString,const std::string toReplace,const std::string replacement)
{

	std::string::size_type n = 0;
	while ( ( n = containingString.find( toReplace, n ) ) != std::string::npos )
	{
	    containingString.replace( n, toReplace.size(), replacement );
	    n += replacement.size();
	}

	return containingString;

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

std::vector<std::string> StringUtil::split(std::string input, std::string regex) {
    // passing -1 as the submatch index parameter performs splitting
	size_t pos = 0;
	std::vector<std::string> result;
	while ((pos = input.find(regex)) != std::string::npos) {
		std::string token;
		token = input.substr(0, pos);
		result.push_back(token);
	  //  std::cout << token << std::endl;
	    input.erase(0, pos + regex.length());
	}
	result.push_back(input);
	return result;
	//std::cout << s << std::endl;

}
