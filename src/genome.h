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

#endif
