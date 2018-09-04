/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
 */

#include "calcite_client.h"

#include <memory>
#include <string>
#include <cstring>
#include <iostream>
#include <thread>

#include <stdio.h>
#include <fcntl.h>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h> //inet_addr
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>

#include "protocol/MessageBuilder.h"

void listenUnixSocket() {
	int unixSocket, clientSocket;
	struct sockaddr_un addr;

	if ((unixSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
		// TODO: Use blazing exception
		std::cerr << "Could not create unix socket" << std::endl;
		exit(EXIT_FAILURE);
	}

	//const Path socketPath = BlazingConfig::getInstance()->getUnixSocketPath();
	const std::string path = "/tmp/fubar.sock";

	std::memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;

	std::strncpy(addr.sun_path, path.c_str(), path.size());
	unlink(path.c_str());

	while (bind(unixSocket, (struct sockaddr*) &addr, sizeof(addr)) == -1) {
		std::cerr << "Could not create unix socket trying again" << std::endl;
		sleep(5);
	}
	std::cout << "bind unix socket done" << std::endl;

	if (listen(unixSocket, 3) == -1) {
		std::cerr << "listen unix socket error" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (;;) {
		clientSocket = accept(unixSocket, nullptr, nullptr);

		if (clientSocket == -1) {
			std::cerr << "accept unix socket error" << std::endl;
			continue;
		}

		std::cout << "Unix socket connection accepted" << std::endl;

		//BlazingThread(&connection_handler, clientSocket, path, true).detach();

		std::cout << "Unix socket handler assigned" << std::endl;
	}
}

//////

//char *socket_path = "./socket";
char *socket_path = "\0hidden";

int a_main() {
	struct sockaddr_un addr;
	char buf[100];
	int fd, rc;

	socket_path = "/tmp/fubar.sock";

	if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
		perror("socket error");
		exit(-1);
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	if (*socket_path == '\0') {
		*addr.sun_path = '\0';
		strncpy(addr.sun_path + 1, socket_path + 1, sizeof(addr.sun_path) - 2);
	} else {
		strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
	}

	if (connect(fd, (struct sockaddr*) &addr, sizeof(addr)) == -1) {
		perror("connect error");
		exit(-1);
	}

	while ((rc = read(STDIN_FILENO, buf, sizeof(buf))) > 0) {
		if (write(fd, buf, rc) != rc) {
			if (rc > 0)
				fprintf(stderr, "partial write");
			else {
				perror("write error");
				exit(-1);
			}
		}
	}

	return 0;
}
//////

void runCalciteClientTest(const std::string &sql) {
	std::cout << "link to calcite service ... %" << std::endl;

	BlazingProtocol::ads_main();

	a_main();
}

a::a() {
	int i = 0;
}
