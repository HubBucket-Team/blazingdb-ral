#include <blazingdb/io/Config/BlazingContext.h>
#include <blazingdb/io/FileSystem/FileSystemRepository.h>

#include "Common/HadoopFileSystemTest.h"
#include <vector>
	
#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Library/Logging/CoutOutput.h>
#include "blazingdb/io/Library/Logging/ServiceLogging.h"



TEST(ConnectTest, ValidConnection) {
	auto output = new Library::Logging::CoutOutput();
	Library::Logging::ServiceLogging::getInstance().setLogOutput(output);

	//auto fileSystemConnection = FileSystemConnection("0.0.0.0", 54310, "aocsa", HadoopFileSystemConnection::DriverType::LIBHDFS3, "");
	const FileSystemConnection fileSystemConnection = SystemEnvironment::getLocalHadoopFileSystemConnection();
	
	const std::unique_ptr<HadoopFileSystem> hadoopFileSystem = std::unique_ptr<HadoopFileSystem>(new HadoopFileSystem(fileSystemConnection));
	
	const std::vector<Uri> uris = hadoopFileSystem->list(Uri("/"));

	for (const Uri &uri : uris) {
		std::cout << (uri.getPath().toString(true)) << std::endl;
	}
	auto path = "hdfs://localhost:54310/Data1Mb/nation_0_0.psv";
	const bool result = hadoopFileSystem->exists(Uri(path));

	FileStatus fileStatus = hadoopFileSystem->getFileStatus(Uri(path));
	EXPECT_TRUE(fileStatus.isFile());
	EXPECT_FALSE(fileStatus.isDirectory());
	EXPECT_EQ(FileType::FILE, fileStatus.getFileType());
	std::cout << fileStatus.getFileSize() << std::endl;



	std::shared_ptr<arrow::io::RandomAccessFile> file = hadoopFileSystem->openReadable(Uri(path));
	ASSERT_NE(nullptr, file);

	int64_t size;

	ASSERT_TRUE(file->GetSize(&size).ok());

	std::shared_ptr<arrow::Buffer> out;
	uint64_t nbytes = fileStatus.getFileSize()/2;

	//EXPECT_EQ(nbytes, size);
	EXPECT_TRUE(file->Read(nbytes, &out).ok());
	EXPECT_EQ(nbytes, out->size());
	std::cout << out->data() << std::endl;
 
	// const std::vector<Uri> uris = hadoopFileSystem->list(Uri("/"));


	//const bool result = hadoopFileSystem->connect(fileSystemConnection);
	EXPECT_TRUE(result);
}


TEST(ConnectTest, ValidConnectionUsingBlazingContext) {
	const Path FS_NAMESPACES_FILE("/tmp/file_system.bin");
	auto output = new Library::Logging::CoutOutput();
	Library::Logging::ServiceLogging::getInstance().setLogOutput(output);

	const FileSystemConnection fileSystemConnection = SystemEnvironment::getLocalHadoopFileSystemConnection();
	for (auto iter  : fileSystemConnection.getConnectionProperties()) {
		std::cout << iter.first << " - " << iter.second << std::endl;
	}
	std::cout << "getFileSystemType: " << (int)fileSystemConnection.getFileSystemType() << std::endl;
	bool result;

	FileSystemEntity fileSystemEntity("tpch_hdfs", fileSystemConnection);
	auto  fileSystemManager = BlazingContext::getInstance()->getFileSystemManager();
	// BlazingContext::getInstance()->getFileSystemManager()->deregisterFileSystem(fileSystemEntity.getAuthority());
	result = BlazingContext::getInstance()->getFileSystemManager()->registerFileSystem(fileSystemEntity);
	if (result) { // then save the fs
		const FileSystemRepository fileSystemRepository(FS_NAMESPACES_FILE, true);
		const bool saved = fileSystemRepository.add(fileSystemEntity);
		if (saved == false) {
			std::cout << "WARNING: could not save the registered file system into ... the data file uri ..."; //TODO percy error message
		}
		const auto all = fileSystemRepository.findAll();
		for (auto fs : all)
			std::cout << "findAll FS: " << fs.getAuthority() << std::endl;
	}
	///user1/warehouse/spark/data/temp/	
	EXPECT_TRUE(result);

	// const std::unique_ptr<HadoopFileSystem> fileSystemManager = std::unique_ptr<HadoopFileSystem>(new HadoopFileSystem(fileSystemConnection));

	const std::vector<Uri> uris =  fileSystemManager->list(Uri("hdfs://tpch_hdfs/"));
	for (const Uri &uri : uris) {
		std::cout << (uri.getPath().toString(true)) << std::endl;
	}
	auto path = "hdfs://tpch_hdfs/Data1Mb/nation_0_0.psv";
	const bool existsParquetDir = fileSystemManager->exists(Uri("hdfs://tpch_hdfs/Data1Mb/"));
	EXPECT_TRUE(existsParquetDir);

	FileStatus fileStatus = fileSystemManager->getFileStatus(Uri(path));
	EXPECT_TRUE(fileStatus.isFile());
	EXPECT_FALSE(fileStatus.isDirectory());
	EXPECT_EQ(FileType::FILE, fileStatus.getFileType());
	std::cout << fileStatus.getFileSize() << std::endl; 

	std::shared_ptr<arrow::io::RandomAccessFile> file = fileSystemManager->openReadable(Uri(path));
	ASSERT_NE(nullptr, file);

	int64_t size;

	ASSERT_TRUE(file->GetSize(&size).ok());

	std::shared_ptr<arrow::Buffer> out;
	uint64_t nbytes = fileStatus.getFileSize()/2;

	//EXPECT_EQ(nbytes, size);
	EXPECT_TRUE(file->Read(nbytes, &out).ok());
	EXPECT_EQ(nbytes, out->size());
	std::cout << out->data() << std::endl;  
}
