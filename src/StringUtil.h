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
    static void findAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr);

    static std::vector<bool> generateQuotedVector(std::string input);
    static int findFirstNotInQuotes(std::string haystack, std::string needle);
    static int findFirstNotInQuotes(std::string haystack, std::string needle, int pos, std::vector<bool> & quoted);
    static int findFirstNotInQuotes(std::string haystack, std::vector<std::string> needles, std::string & needleFound);
    static int findFirstNotInQuotes(std::string haystack, std::vector<std::string> needles, std::string & needleFound, int startPos, std::vector<bool> & quoted);
};

#endif /* STRINGUTIL_H_ */
