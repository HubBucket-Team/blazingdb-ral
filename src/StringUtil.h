/*
 * StringUtil.h
 *
 *  Created on: Aug 7, 2014
 *      Author: felipe
 */

#ifndef STRINGUTIL_H_
#define STRINGUTIL_H_

#include <vector>
#include <string>

class StringUtil {
public:
	StringUtil();
	virtual ~StringUtil();
    static std::vector<std::string> split(std::string &s, char delim);
    static std::vector<std::string> & split(std::string &s, char delim, std::vector<std::string> &elems);
    static std::string & rtrim(std::string &s);
	static std::string replace(std::string containingString,const std::string toReplace,const std::string replacement);
    static std::vector<std::string> split(std::string input, std::string regex);
};

#endif /* STRINGUTIL_H_ */
