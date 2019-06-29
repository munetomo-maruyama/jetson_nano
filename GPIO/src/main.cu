//====================================================
// GPIO Control
//     main.cu : Main Routine
//----------------------------------------------------
// Rev.01 2019.06.08 M.Munetomo
//----------------------------------------------------
// Copyright (C) 2019 Munetomo Maruyama
//====================================================

#include <cinttypes>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//----------------------------------
// Main Routine
//----------------------------------
int main(void)
{
	int fd, fd_poll;
	int byte;
	
	// Teach how to quit
	printf("Please Hit Enter to Quit.\n");
	
    // Export GPIO50 (11pin)
    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "50", 3);
    close(fd);

    // Set Direction of GPIO50 (11pin) as Input
    fd = open("/sys/class/gpio/gpio50/direction", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "in", 3);
    close(fd);

    // Set Edge Detection of GPIO50 (11pin) as Both
    fd = open("/sys/class/gpio/gpio50/edge", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "both", 5);
    close(fd);
    
    // Left Open Value of GPIO50 (11pin)
    fd_poll = open("/sys/class/gpio/gpio50/value", O_RDONLY | O_NONBLOCK);
    if (fd_poll < 0) exit(EXIT_FAILURE);

    // Export GPIO79 (12pin)
    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "79", 3);
    close(fd);

    // Set Direction of GPIO79 (12pin) as Output
    fd = open("/sys/class/gpio/gpio79/direction", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "out", 4);
    close(fd);
    
    // Prepare Interrupt Polling
    struct pollfd fdset[2];
    int    nfds = 2;
    memset((void*)fdset, 0, sizeof(fdset));
    fdset[0].fd = STDIN_FILENO;
    fdset[0].events = POLLIN;
    fdset[1].fd = fd_poll;
    fdset[1].events = POLLPRI;

    // Forever Loop
    while(1)
    {
		char ch;
		
		// Wait for Interrupt
		int rc = poll(fdset, nfds, 3000); // polling with timeout 3sec
		if (rc < 0) exit(EXIT_FAILURE);
		// Timeout?
		if (rc == 0)
		{
			printf("Timeout\n");
			continue;
		}
		// Detected Stdin?
        if (fdset[0].revents & POLLIN)
        {
			byte = read(fdset[0].fd, &ch, 1);
			printf("poll() stdin read %c\n", ch);
			break;
		}
		// Detected GPIO50 Edge?
        if (fdset[1].revents & POLLPRI)
        {
            lseek(fdset[1].fd, 0, SEEK_SET);
			byte = read(fdset[1].fd, &ch, 1);
			printf("poll() GPIO50 Interrupt Detected, level=%c\n", ch);
			//
			// Toggle GPIO79 (12pin) 10times
            for (int i = 0; i < 10; i++)
            {
				// Open
                fd = open("/sys/class/gpio/gpio79/value", O_WRONLY);
                if (fd < 0) exit(EXIT_FAILURE);
		        // Set High
                byte = write(fd, "1", 2);
                // Wait
                usleep(1000);
		        // Set Low
                byte = write(fd, "0", 2);
                // Wait
                usleep(1000);
                // Close
                close(fd);
			}
		}
	}
    
    // Close File of Value of GPIO50 (11pin)
    close(fd_poll);

    // UnExport GPIO50 (11pin)
    fd = open("/sys/class/gpio/unexport", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "50", 3);
    close(fd);

    // UnExport GPIO79 (12pin)
    fd = open("/sys/class/gpio/unexport", O_WRONLY);
    if (fd < 0) exit(EXIT_FAILURE);
    byte = write(fd, "79", 3);
    close(fd);

    // Return from this Program
    return(EXIT_SUCCESS);
}

//====================================================
// End of Program
//====================================================
