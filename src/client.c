/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files ((the "LICENSE")), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Code directly taken from https://www.geeksforgeeks.org/tcp-server-client-implementation-in-c/ and modified
#include "llab.h"

void contact_server(int sockfd, int buffer_size, int reading_pipe, int writing_pipe) { 
    float* buff = (float*)calloc(buffer_size,sizeof(float));
    int ret;
    while(1){   
        while(read(sockfd, buff, sizeof(float)*buffer_size) == 0);// waiting for server
        ret = write(writing_pipe,buff, sizeof(float)*buffer_size);// writing to parent process
        while(read(reading_pipe, buff, sizeof(float)*buffer_size) == 0);// waiting for parent process
        ret = write(sockfd,buff, sizeof(float)*buffer_size);// writing to server
        
    }
    free(buff);
} 


/* See server.c for more specific details
 * 
 * 
 * Inputs:
 *             
 *                 @ int port:= the port of the server
 *                 @ char* server_address:= the server address
 *                 @ int buffer_size:= the buffer size written by the server and parent process
 *                 @ int reading_pipe:= to read from parent
 *                 @ int writing_pipe:= to write to parent
 * */
int run_client(int port, char* server_address, int buffer_size, int reading_pipe, int writing_pipe){
    
    int sockfd, connfd; 
    struct sockaddr_in servaddr, cli;
    int sockaddr_len = sizeof(struct sockaddr_in);
  
    // socket create and varification 
    sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    if (sockfd == -1) { 
        fprintf(stderr,"Error: socket creation failed...\n"); 
        exit(1); 
    } 
    else
        printf("Socket successfully created..\n"); 
        
    bzero(&servaddr, sizeof(servaddr)); 
  
    // assign IP, PORT 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_addr.s_addr = inet_addr(server_address); 
    servaddr.sin_port = htons(port); 
  
    // connect the client socket to server socket 
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) { 
        printf("connection with the server failed...\n"); 
        exit(0); 
    } 
    else
        printf("connected to the server..\n");  
  
    // function for chat 
    contact_server(sockfd,buffer_size,reading_pipe,writing_pipe); 
  
    // close the socket 
    close(sockfd);
    
    return 0;
    
}

