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

#include "llab.h"
void* server_thread(void* _args) {
    
    // depacking args
    thread_args_server* args = (thread_args_server*) _args;
    float* buff = (float*)calloc(args->buffer_size,sizeof(float));
    int ret;
    while(1){
        ret = 0;
        while(ret == 0){
            ret = read(args->reading_pipe, buff, sizeof(float)*args->buffer_size);
            if(ret == -1){
                printf("Error description is : %s\n",strerror(errno));
                exit(0);
            }
        }// waiting for parent process
        ret = write(args->client_desc,buff, sizeof(float)*args->buffer_size);// writing to client
        if(ret == -1){
            printf("Error description is : %s\n",strerror(errno));
            exit(0);
        }
        ret = 0;
        
        while(ret == 0){
            if(sizeof(float)*args->buffer_size <= 4096){
                ret = read(args->client_desc, buff, sizeof(float)*args->buffer_size);
                if(ret == -1){
                    printf("Error description is : %s\n",strerror(errno));
                    exit(0);
                }
            }
            else{
                long long unsigned int sum = 0;
                char buffer[sizeof(float)*args->buffer_size];
                ret = read(args->client_desc, buffer, 4096);
                if(ret == -1){
                    printf("Error description is : %s\n",strerror(errno));
                    exit(0);
                }
                sum+=ret;
                long long unsigned int count;
                for(;sum < sizeof(float)*args->buffer_size;  sum+=ret){
                    ret = 0;
                    if(sizeof(float)*args->buffer_size-sum < 4096){
                        ret = read(args->client_desc, &buffer[sum], sizeof(float)*args->buffer_size-sum);
                        if(ret == -1){
                            printf("Error description is : %s\n",strerror(errno));
                            exit(0);
                        }
                    }
                    else{
                        ret = read(args->client_desc, &buffer[sum], 4096);
                        if(ret == -1){
                            printf("Error description is : %s\n",strerror(errno));
                            exit(0);
                        }
                    }
                }
                memcpy(buff,buffer,args->buffer_size*sizeof(float));
            }
        }// waiting for client
        ret = write(args->writing_pipe,buff, sizeof(float)*args->buffer_size);// writing to parent process
        if(ret == -1){
            printf("Error description is : %s\n",strerror(errno));
            exit(0);
        }
    }
    
    free(buff);
    
    return NULL;
}


/* This function creates a server on your current ip address on the port: port,
 * accept a maximum number of connections = max_num_conn, and create each thread per connection.
 * each thread read a float vector from a client and writes on the writing_pipes[i] this vector, then
 * wait for a float vector from reading_pipe[i] and send this vector to the client.
 * The ideal situation is: client -> compute some instances of the mini batch, send to server the partial derivatives
 * the thread of the client on the server side read these partial derivatives, send back to the father process with writing pipes,
 * the parent process sum up all these partial derivatives, update the model, send back to the thread the model updated, the thread
 * send to the client and the client goes on and compute again the new partial derivatives, and so on...
 * 
 * Inputs:
 * 
 *                 @ int port:= the server port
 *                 @ int max_num_conn:= the maximum number of connections accepted by the server
 *                 @ int* reading_pipes:= a pipe writing for float vector to the thread
 *                 @ int* writing_pipes:= where the thread write its vector
 *                 @ int buffer_size:= the buffer of the vector
 * 
 * */
int run_server(int port, int max_num_conn, int* reading_pipes, int* writing_pipes, int buffer_size, char* ip){
    int socket_desc, client_desc,sockaddr_len = sizeof(struct sockaddr_in),c = 1;
    int ret;
    struct sockaddr_in server_addr;
    struct sockaddr_in* client_addr;
    struct sockaddr_in* client_addr2;
    
    bzero(&server_addr, sizeof(server_addr)); 
    inet_pton(AF_INET, ip, &(&server_addr)->sin_addr);
    // socket creation
    socket_desc = socket(AF_INET,SOCK_STREAM,0);
    if(socket_desc == -1){
        fprintf(stderr,"Error: can't create socket\n");
        exit(1);
    }
    
    // Which connection can accept
    
    server_addr.sin_addr.s_addr = INADDR_ANY;
    
    // Ip family
    server_addr.sin_family = AF_INET;
    // Port
    
    server_addr.sin_port = htons(port);
    
    // Handling crash case to reuse the descriptor
    
    ret = setsockopt(socket_desc,SOL_SOCKET,SO_REUSEADDR,&c,sizeof(int));
    if(ret == -1){
        fprintf(stderr,"Error: setsockopt failed..\n");
        exit(1);
    }
    
    // Binding newly created socket to given IP and verification 
    if ((bind(socket_desc, (struct sockaddr *)&server_addr, sizeof(server_addr))) != 0) { 
        printf("socket bind failed...\n"); 
        exit(0); 
    } 
    else
        printf("Socket successfully binded..\n"); 
    
    // listen on tot number of connections
    ret = listen(socket_desc,max_num_conn);
    if(ret == -1){
        fprintf(stderr,"Error, listen failed\n");
        exit(1);
    }
    int i = 0;
    while(1){
        client_addr = calloc(1,sizeof(struct sockaddr_in));
        client_desc = accept(socket_desc,(struct sockaddr*)&client_addr,(socklen_t*)&sockaddr_len);
        if((client_desc == 1 && errno == EINTR)) continue;
        thread_args_server* thread = (thread_args_server*)malloc(sizeof(thread_args_server));
        thread->idx = i;
        thread->client_desc = client_desc;
        thread->client_addr = client_addr;
        thread->reading_pipe = reading_pipes[i];
        thread->writing_pipe = writing_pipes[i];
        thread->buffer_size = buffer_size;
        pthread_t t;
        pthread_create(&t,NULL,server_thread,thread);
        pthread_detach(t);
        printf("connected client id: %d\n",i);
        i++;
    }
    
    pthread_exit(NULL);
    return 0;
    
}
