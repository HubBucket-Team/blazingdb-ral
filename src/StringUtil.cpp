/*
 * StringUtil.cpp
 *
 *  Created on: Aug 7, 2014
 *      Author: felipe
 */

#include "StringUtil.h"
#include <sstream>
#include <algorithm>
#include <limits.h>

StringUtil::StringUtil() {
	// TODO Auto-generated constructor stub
}

StringUtil::~StringUtil() {
	// TODO Auto-generated destructor stub
}
#include <iostream>
void StringUtil::findAndReplaceAll(std::string & data, std::string toSearch, std::string replaceStr)
{
	// Get the first occurrence
	size_t pos = data.find(toSearch);

	// Repeat till end is reached
	while( pos != std::string::npos)
	{
		// Replace this occurrence of Sub String
		data.replace(pos, toSearch.size(), replaceStr);
		// Get the next occurrence from the current position
		pos =data.find(toSearch, pos + toSearch.size());
	}
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


std::vector<bool> StringUtil::generateQuotedVector(std::string input){
	std::vector<bool> quoted(input.size());
	bool inQuote = false;
	bool inDoubleQuote = false;
	int i = 0;
	if (input.size() > 2){
		if (input[i]=='\'' && !(input[i+1]=='\'')){
			inQuote = !inQuote;
		}
		if (input[i]=='"'){
			inDoubleQuote = !inDoubleQuote;
		}
		quoted[i]=inQuote | inDoubleQuote;
		for (int i=1; i<quoted.size()-1; i++){
			if (input[i]=='\'' && !(input[i+1]=='\'' || input[i-1]=='\'')){
				inQuote = !inQuote;
			}
			if (input[i]=='"'){
				inDoubleQuote = !inDoubleQuote;
			}
			quoted[i]=inQuote | inDoubleQuote;
		}
		i = quoted.size()-1;
		if (input[i]=='\'' && !(input[i-1]=='\'')){
			inQuote = !inQuote;
		}
		if (input[i]=='"'){
			inDoubleQuote = !inDoubleQuote;
		}
		quoted[i]=inQuote | inDoubleQuote;
	}
	return quoted;
}

int StringUtil::findFirstNotInQuotes(std::string haystack, std::string needle) {
	std::vector<bool> quoted = generateQuotedVector(haystack);
	return findFirstNotInQuotes(haystack, needle, 0, quoted);
}

int StringUtil::findFirstNotInQuotes(std::string haystack, std::vector<std::string> needles, std::string & needleFound) {
	std::vector<bool> quoted = generateQuotedVector(haystack);
	return findFirstNotInQuotes(haystack, needles, needleFound, 0, quoted);
}

int StringUtil::findFirstNotInQuotes(std::string haystack, std::string needle, int pos, std::vector<bool> & quoted) {

	if (quoted.size() != haystack.size()){
		quoted = generateQuotedVector(haystack);
	}

	while (pos < haystack.size()){
		pos = haystack.find(needle, pos);
		if (pos == -1 || !quoted[pos]){
			return pos;
		}
		pos += needle.size();
	}
	return -1;
}

int StringUtil::findFirstNotInQuotes(std::string haystack, std::vector<std::string> needles, std::string & needleFound, int startPos, std::vector<bool> & quoted) {

	if (quoted.size() != haystack.size()){
		quoted = generateQuotedVector(haystack);
	}

	int minPos = INT_MAX;
	needleFound = "";
	for (int i = 0; i < needles.size(); i++){
		int pos = startPos;
		while (pos != -1){
			pos = haystack.find(needles[i], pos);
			if (pos != -1){
				if (!quoted[pos] && pos < minPos){
					minPos = pos;
					needleFound = needles[i];
					break;
				} else {
					pos = pos + needles[i].size();
				}
			}
		}
	}
	if (minPos == INT_MAX){
		return -1;
	} else {
		return minPos;
	}
}
