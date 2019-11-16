#ifndef __FEED_STRUCTURE_H__
#define __FEED_STRUCTURE_H__
#include "genome.h"

typedef struct ff{
    node** list_nodes;
    connection** list_connections;
    int size,flag;// size = size of list_nodes, flag = 0 list node ends with input, 1 with loop, -1 with not loop not input
}ff;


#endif
