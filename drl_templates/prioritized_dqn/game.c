#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define FRAME_SIZE 32//can be changed
#define NUMBER_FRAMES 3//can be changed
#define CHANNELS 3 // can be changed

void game_run(int fd1, int fd2,int fd3){
    
    int i;
    
    float* message = (float*)calloc(FRAME_SIZE*NUMBER_FRAMES*CHANNELS,sizeof(float));//can be changed
    float reward = 0;
    
    i = write(fd2,message,sizeof(float)*FRAME_SIZE*NUMBER_FRAMES*CHANNELS);//write beginning frames
    i = write(fd3,&reward,sizeof(float));//write beginning reward
    int flag = 1,cmd;
    
    while(flag){
        while(read(fd1,&cmd,sizeof(int))==0);//waiting for action
        // Do stuff
        // Get new frames and new reward
        // if game ended reward = -1, flag = 0
        i = write(fd2,message,sizeof(float)*FRAME_SIZE*NUMBER_FRAMES*CHANNELS);//writeframes
        i = write(fd3,&reward,sizeof(float));//write reward
    }
    
}
