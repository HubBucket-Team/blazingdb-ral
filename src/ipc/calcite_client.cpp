/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
 */

#include "calcite_client.h"

#include <memory>
#include <string>
#include <cstring>
#include <iostream>

#include <stdio.h>
#include <fcntl.h>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h> //inet_addr

void listenUnixSocket() {
	int unixSocket, clientSocket;
	struct sockaddr_un addr;

	if ((unixSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
		// TODO: Use blazing exception
		std::cerr << "Could not create unix socket" << std::endl;
		exit (EXIT_FAILURE);
	}

	//const Path socketPath = BlazingConfig::getInstance()->getUnixSocketPath();
	const std::string path = "/tmp/hola.sock";

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
		exit (EXIT_FAILURE);
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

a::a() {
	int i = 0;
}
