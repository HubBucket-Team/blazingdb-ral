#include "SystemEnvironment.h"

#include <cstdlib>

#include "FileSystem/HadoopFileSystem.h"

namespace SystemEnvironment {

std::string getConnectionPropertyEnvValue(const std::string &connectionPropertyEnvName) {
	const char *envValue = std::getenv(connectionPropertyEnvName.c_str());
	const bool isDefined = (envValue != nullptr);

	if (isDefined == false) {
		const std::string error = "FATAL: You need to define the environment variable: " + connectionPropertyEnvName;
		throw std::invalid_argument(error);
	}
	// std::map<std::string, std::string> var_envs;
	// var_envs["BLAZING_HDFS_HOST"]="localhost";
	// var_envs["BLAZING_HDFS_PORT"]="54310";
	// var_envs["BLAZING_HDFS_USER"]="aocsa";
	// var_envs["BLAZING_HDFS_DRIVER_TYPE"]="LIBHDFS3";
	// var_envs["BLAZING_HDFS_KERBEROS_TICKET"]="";

	const std::string propertyEnvValue = isDefined ? std::string(envValue) : std::string();

	return propertyEnvValue;
}

std::string getConnectionPropertyEnvValue(ConnectionProperty connectionProperty) {
	const std::string propertyEnvName = connectionPropertyEnvName(connectionProperty);
	return getConnectionPropertyEnvValue(propertyEnvName);
}

const std::string getHostEnvValue() {
	const std::string value = getConnectionPropertyEnvValue(ConnectionProperty::HOST);
	return value;
}

const int getPortEnvValue() {
	const std::string value = getConnectionPropertyEnvValue(ConnectionProperty::PORT);
	const int port = atoi(value.c_str());
	return port;
}

const std::string getUserEnvValue() {
	const std::string value = getConnectionPropertyEnvValue(ConnectionProperty::USER);
	return value;
}

const DriverType getDriverTypeEnvValue() {
	const std::string value = getConnectionPropertyEnvValue(ConnectionProperty::DRIVER_TYPE);
	const DriverType driverType = driverTypeFromName(value);
	return driverType;
}

const std::string getkerberosTicketEnvValue() {
	const std::string value = getConnectionPropertyEnvValue(ConnectionProperty::KERBEROS_TICKET);
	return value;
}

const FileSystemConnection getLocalHadoopFileSystemConnection() {
	const std::string host = SystemEnvironment::getHostEnvValue();
	const int port = SystemEnvironment::getPortEnvValue();
	const std::string user = SystemEnvironment::getUserEnvValue();
	const DriverType driverType = SystemEnvironment::getDriverTypeEnvValue();
	const std::string kerberosTicket = SystemEnvironment::getkerberosTicketEnvValue();
	const FileSystemConnection fileSystemConnection(host, port, user, driverType, kerberosTicket);

	return fileSystemConnection;
}

}
