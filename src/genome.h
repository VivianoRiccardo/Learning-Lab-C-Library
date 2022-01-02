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


#ifndef __GENOME_H__
#define __GENOME_H__


typedef struct node{
    struct connection** in_connections;
    struct connection** out_connections;
    int innovation_number,in_conn_size,out_conn_size,flag,bias_flag; //the flag is used for the feed forward
    float actual_value, stored_value;
    
}node;

typedef struct connection{
    struct node* in_node;
    struct node* out_node;
    int innovation_number,flag; //flag = 0 is not used the connection
    float weight;
}connection;

typedef struct genome{
    struct node** all_nodes; /*the first nodes are inputs and outputs*/ 
    int number_input,number_output, number_total_nodes,specie_rip;
    float fitness;
}genome;

typedef struct thread_args_genome {
    genome* g;
    float* input;
    float** output;
    int global_inn_numb_nodes;
    int global_inn_numb_connections;
    int index;
} thread_args_genome;

#endif
