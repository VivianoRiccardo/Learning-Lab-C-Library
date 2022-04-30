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


int round_up(float num) { 
    if(num!=(int)num)
        return (int)((int)num+1);
    return num; 
}


int save_genome(genome* g, int global_inn_numb_connections, int numb){
    
    int i,n;
    connection** c = get_connections(g,global_inn_numb_connections);
    n = get_numb_connections(g,global_inn_numb_connections);
    char string[20];
    char *s = ".bin";
    FILE *write_ptr;
    
    itoa(numb, string);
    
    strcat(string,s);
    
    write_ptr = fopen(string,"wb");
    
    convert_data(&g->number_input,sizeof(int),1);
    fwrite(&g->number_input,sizeof(int),1,write_ptr);
    convert_data(&g->number_input,sizeof(int),1);
    convert_data(&g->number_output,sizeof(int),1);
    fwrite(&g->number_output,sizeof(int),1,write_ptr);
    convert_data(&g->number_output,sizeof(int),1);
    convert_data(&g->number_total_nodes,sizeof(int),1);
    fwrite(&g->number_total_nodes,sizeof(int),1,write_ptr);
    convert_data(&g->number_total_nodes,sizeof(int),1);
    convert_data(&g->fitness,sizeof(float),1);
    fwrite(&g->fitness,sizeof(float),1,write_ptr);
    convert_data(&g->fitness,sizeof(float),1);
    
    for(i = 0; i < g->number_total_nodes; i++){
        convert_data(&g->all_nodes[i]->in_conn_size,sizeof(int),1);
        fwrite(&g->all_nodes[i]->in_conn_size,sizeof(int),1,write_ptr);
        convert_data(&g->all_nodes[i]->in_conn_size,sizeof(int),1);
        convert_data(&g->all_nodes[i]->out_conn_size,sizeof(int),1);
        fwrite(&g->all_nodes[i]->out_conn_size,sizeof(int),1,write_ptr);
        convert_data(&g->all_nodes[i]->out_conn_size,sizeof(int),1);
        convert_data(&g->all_nodes[i]->innovation_number,sizeof(int),1);
        fwrite(&g->all_nodes[i]->innovation_number,sizeof(int),1,write_ptr);
        convert_data(&g->all_nodes[i]->innovation_number,sizeof(int),1);
    }
    convert_data(&n,sizeof(int),1);
    fwrite(&n,sizeof(int),1,write_ptr);
    convert_data(&n,sizeof(int),1);
    
    for(i = 0; i < n; i++){
        convert_data(&c[i]->innovation_number,sizeof(int),1);
        fwrite(&c[i]->innovation_number,sizeof(int),1,write_ptr);
        convert_data(&c[i]->innovation_number,sizeof(int),1);
        convert_data(&c[i]->in_node->innovation_number,sizeof(int),1);
        fwrite(&c[i]->in_node->innovation_number,sizeof(int),1,write_ptr);
        convert_data(&c[i]->in_node->innovation_number,sizeof(int),1);
        convert_data(&c[i]->out_node->innovation_number,sizeof(int),1);
        fwrite(&c[i]->out_node->innovation_number,sizeof(int),1,write_ptr);
        convert_data(&c[i]->out_node->innovation_number,sizeof(int),1);
        convert_data(&c[i]->weight,sizeof(float),1);
        fwrite(&c[i]->weight,sizeof(float),1,write_ptr);
        convert_data(&c[i]->weight,sizeof(float),1);
        convert_data(&c[i]->flag,sizeof(int),1);
        fwrite(&c[i]->flag,sizeof(int),1,write_ptr);
        convert_data(&c[i]->flag,sizeof(int),1);
    }
    
    free(c);
    i = fclose(write_ptr);
    
    return i;
    
}

char* get_genome_array(genome* g, int global_inn_numb_connections){
    int i,n, sum=0;
    connection** cc = get_connections(g,global_inn_numb_connections);
    n = get_numb_connections(g,global_inn_numb_connections);
    
    char* c = (char*)malloc(sizeof(int)*3 + sizeof(int)*3*g->number_total_nodes + sizeof(int) + sizeof(int)*4*n + sizeof(float)+ sizeof(float)*n);
    convert_data(&(g->number_input),sizeof(int),1);
    memcpy(c+sum,&(g->number_input),sizeof(int));
    convert_data(&(g->number_input),sizeof(int),1);
    sum+=sizeof(int);
    convert_data(&(g->number_output),sizeof(int),1);
    memcpy(c+sum,&(g->number_output),sizeof(int));
    convert_data(&(g->number_output),sizeof(int),1);
    sum+=sizeof(int);
    convert_data(&(g->number_total_nodes),sizeof(int),1);
    memcpy(c+sum,&(g->number_total_nodes),sizeof(int));
    convert_data(&(g->number_total_nodes),sizeof(int),1);
    sum+=sizeof(int);
    convert_data(&(g->fitness),sizeof(float),1);
    memcpy(c+sum,&(g->fitness),sizeof(float));
    convert_data(&(g->fitness),sizeof(float),1);
    sum+=sizeof(float);
    
    for(i = 0; i < g->number_total_nodes; i++){
        convert_data(&(g->all_nodes[i]->in_conn_size),sizeof(int),1);
        memcpy(c+sum,&(g->all_nodes[i]->in_conn_size),sizeof(int));
        convert_data(&(g->all_nodes[i]->in_conn_size),sizeof(int),1);
        sum+=sizeof(int);
        convert_data(&(g->all_nodes[i]->in_conn_size),sizeof(int),1);
        memcpy(c+sum,&(g->all_nodes[i]->out_conn_size),sizeof(int));
        convert_data(&(g->all_nodes[i]->in_conn_size),sizeof(int),1);
        sum+=sizeof(int);
        convert_data(&(g->all_nodes[i]->innovation_number),sizeof(int),1);
        memcpy(c+sum,&(g->all_nodes[i]->innovation_number),sizeof(int));
        convert_data(&(g->all_nodes[i]->innovation_number),sizeof(int),1);
        sum+=sizeof(int);
    }
    convert_data(&(n),sizeof(int),1);
    memcpy(c+sum,&(n),sizeof(int));
    convert_data(&(n),sizeof(int),1);
    sum+=sizeof(int);
    
    
    for(i = 0; i < n; i++){
        convert_data(&(cc[i]->innovation_number),sizeof(int),1);
        memcpy(c+sum,&(cc[i]->innovation_number),sizeof(int));
        convert_data(&(cc[i]->innovation_number),sizeof(int),1);
        sum+=sizeof(int);
        convert_data(&(cc[i]->in_node->innovation_number),sizeof(int),1);
        memcpy(c+sum,&(cc[i]->in_node->innovation_number),sizeof(int));
        convert_data(&(cc[i]->in_node->innovation_number),sizeof(int),1);
        sum+=sizeof(int);
        convert_data(&(cc[i]->out_node->innovation_number),sizeof(int),1);
        memcpy(c+sum,&(cc[i]->out_node->innovation_number),sizeof(int));
        convert_data(&(cc[i]->out_node->innovation_number),sizeof(int),1);
        sum+=sizeof(int);
        convert_data(&(cc[i]->weight),sizeof(float),1);
        memcpy(c+sum,&(cc[i]->weight),sizeof(float));
        convert_data(&(cc[i]->weight),sizeof(float),1);
        sum+=sizeof(float);
        convert_data(&(cc[i]->flag),sizeof(int),1);
        memcpy(c+sum,&(cc[i]->flag),sizeof(int));
        convert_data(&(cc[i]->flag),sizeof(int),1);
        sum+=sizeof(int);
    }
    
    free(cc);    
    return c;
}


int get_genome_array_size(genome* g, int global_inn_numb_connections){
    int i,n;
    connection** cc = get_connections(g,global_inn_numb_connections);
    n = get_numb_connections(g,global_inn_numb_connections);
    free(cc);
    return sizeof(int)*3 + sizeof(int)*3*g->number_total_nodes + sizeof(int) + sizeof(int)*4*n + sizeof(float)+ sizeof(float)*n;
}

genome* init_genome_from_array(int global_inn_numb_connections, char* g_array){
    int i,j,n,inn,inn2,k, sum=0;
    char input[256];

    connection** c = (connection**)malloc(sizeof(connection*)*global_inn_numb_connections);
    for(i = 0; i < global_inn_numb_connections; i++){
        c[i] = NULL;
    }
    genome* g = (genome*)malloc(sizeof(genome));
    
    memcpy(&(g->number_input),g_array+sum,sizeof(int));
    convert_data(&(g->number_input),sizeof(int),1);
    sum+=sizeof(int);
    memcpy(&(g->number_output),g_array+sum,sizeof(int));
    convert_data(&(g->number_output),sizeof(int),1);
    sum+=sizeof(int);
    memcpy(&(g->number_total_nodes),g_array+sum,sizeof(int));
    convert_data(&(g->number_total_nodes),sizeof(int),1);
    sum+=sizeof(int);
    memcpy(&(g->fitness),g_array+sum,sizeof(float));
    convert_data(&(g->fitness),sizeof(float),1);
    sum+=sizeof(float);
    g->all_nodes = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    
    for(i = 0; i < g->number_total_nodes; i++){
        g->all_nodes[i] = (node*)malloc(sizeof(node));
        memcpy(&(g->all_nodes[i]->in_conn_size),g_array+sum,sizeof(int));
        convert_data(&(g->all_nodes[i]->in_conn_size),sizeof(int),1);
        sum+=sizeof(int);
        memcpy(&(g->all_nodes[i]->out_conn_size),g_array+sum,sizeof(int));
        convert_data(&(g->all_nodes[i]->out_conn_size),sizeof(int),1);
        sum+=sizeof(int);
        memcpy(&(g->all_nodes[i]->innovation_number),g_array+sum,sizeof(int));
        convert_data(&(g->all_nodes[i]->innovation_number),sizeof(int),1);
        sum+=sizeof(int);
        g->all_nodes[i]->actual_value = 0;
        g->all_nodes[i]->stored_value = 0;
        g->all_nodes[i]->in_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->in_conn_size);
        g->all_nodes[i]->out_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
    }
    
    memcpy(&n,g_array+sum,sizeof(int));
    convert_data(&n,sizeof(int),1);
    sum+=sizeof(int);
    
    for(i = 0; i < n; i++){
        memcpy(&(inn),g_array+sum,sizeof(int));
        convert_data(&(inn),sizeof(int),1);
        sum+=sizeof(int);
        free(c[inn-1]);
        c[inn-1] = (connection*)malloc(sizeof(connection));
        c[inn-1]->innovation_number = inn;
        memcpy(&(inn2),g_array+sum,sizeof(int));
        convert_data(&(inn2),sizeof(int),1);
        sum+=sizeof(int);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2)
                c[inn-1]->in_node = g->all_nodes[j];
        }
        memcpy(&(inn2),g_array+sum,sizeof(int));
        convert_data(&(inn2),sizeof(int),1);
        sum+=sizeof(int);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2)
                c[inn-1]->out_node = g->all_nodes[j];
        }
        memcpy(&c[inn-1]->weight,g_array+sum,sizeof(float));
        convert_data(&c[inn-1]->weight,sizeof(float),1);
        sum+=sizeof(float);
        memcpy(&c[inn-1]->flag,g_array+sum,sizeof(int));
        convert_data(&c[inn-1]->flag,sizeof(int),1);
        sum+=sizeof(int);
    }
    
    for(i = 0; i < g->number_total_nodes; i++){
        inn = 0;
        inn2 = 0;
        for(j = 0; j < global_inn_numb_connections; j++){
            if(c[j]!=NULL){
                if(c[j]->in_node->innovation_number == g->all_nodes[i]->innovation_number){
                    g->all_nodes[i]->out_connections[inn] = c[j];
                    inn++;
                }
                if(c[j]->out_node->innovation_number == g->all_nodes[i]->innovation_number){

                    g->all_nodes[i]->in_connections[inn2] = c[j];
                    inn2++;
                }
                
            }
        }
            
        
    }
    
    for(j = 0; j < global_inn_numb_connections; j++){
        if(c[j] == NULL)
            free(c[j]);
    }
    
    free(c);
    
    return g;
    
    
}

genome* load_genome(int global_inn_numb_connections, char* filename){
    int i,j,n,inn,inn2,k;
    char input[256];
    FILE *read_ptr = fopen(filename,"r");
    if(read_ptr == NULL){
        fprintf(stderr,"Error no such a file\n");
        exit(1);
    }
    
    connection** c = (connection**)malloc(sizeof(connection*)*global_inn_numb_connections);
    for(i = 0; i < global_inn_numb_connections; i++){
        c[i] = NULL;
    }
    genome* g = (genome*)malloc(sizeof(genome));
    
    k = fread(&g->number_input,sizeof(int),1,read_ptr);
    convert_data(&g->number_input,sizeof(int),1);
    k = fread(&g->number_output,sizeof(int),1,read_ptr);
    convert_data(&g->number_output,sizeof(int),1);
    k = fread(&g->number_total_nodes,sizeof(int),1,read_ptr);
    convert_data(&g->number_total_nodes,sizeof(int),1);
    k = fread(&g->fitness,sizeof(float),1,read_ptr);
    convert_data(&g->fitness,sizeof(float),1);
    g->all_nodes = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    
    for(i = 0; i < g->number_total_nodes; i++){
        g->all_nodes[i] = (node*)malloc(sizeof(node));
        k = fread(&g->all_nodes[i]->in_conn_size,sizeof(int),1,read_ptr);
        convert_data(&g->all_nodes[i]->in_conn_size,sizeof(int),1);
        k = fread(&g->all_nodes[i]->out_conn_size,sizeof(int),1,read_ptr);
        convert_data(&g->all_nodes[i]->out_conn_size,sizeof(int),1);
        k = fread(&g->all_nodes[i]->innovation_number,sizeof(int),1,read_ptr);
        convert_data(&g->all_nodes[i]->innovation_number,sizeof(int),1);
        g->all_nodes[i]->actual_value = 0;
        g->all_nodes[i]->stored_value = 0;
        g->all_nodes[i]->in_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->in_conn_size);
        g->all_nodes[i]->out_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
    }
    
    k = fread(&n,sizeof(int),1,read_ptr);
    convert_data(&n,sizeof(int),1);
    for(i = 0; i < n; i++){
        k = fread(&inn,sizeof(int),1,read_ptr);
        convert_data(&inn,sizeof(int),1);
        free(c[inn-1]);
        c[inn-1] = (connection*)malloc(sizeof(connection));
        c[inn-1]->innovation_number = inn;
        k = fread(&inn2,sizeof(int),1,read_ptr);
        convert_data(&inn2,sizeof(int),1);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2){
                c[inn-1]->in_node = g->all_nodes[j];
                break;
            }
        }
        k = fread(&inn2,sizeof(int),1,read_ptr);
        convert_data(&inn2,sizeof(int),1);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2){
                c[inn-1]->out_node = g->all_nodes[j];
                break;
            }
        }
        
        k = fread(&c[inn-1]->weight,sizeof(float),1,read_ptr);
        convert_data(&c[inn-1]->weight,sizeof(float),1);
        k = fread(&c[inn-1]->flag,sizeof(int),1,read_ptr);
        convert_data(&c[inn-1]->flag,sizeof(int),1);
    }
    
    for(i = 0; i < g->number_total_nodes; i++){
        inn = 0;
        inn2 = 0;
        for(j = 0; j < global_inn_numb_connections; j++){
            if(c[j]!=NULL){
                if(c[j]->in_node->innovation_number == g->all_nodes[i]->innovation_number){
                    g->all_nodes[i]->out_connections[inn] = c[j];
                    inn++;
                }
                if(c[j]->out_node->innovation_number == g->all_nodes[i]->innovation_number){

                    g->all_nodes[i]->in_connections[inn2] = c[j];
                    inn2++;
                }
                
            }
        }
            
        
    }
    
    for(j = 0; j < global_inn_numb_connections; j++){
        if(c[j] == NULL)
            free(c[j]);
    }
    
    free(c);
    
    i = fclose(read_ptr);
    
    
    if(i == EOF){
        printf("error closing the file, the process will end\n");
        exit(1);
    }
    
    return g;
    
    
}

float random_float_number(float a){
    float x = (float)rand()/(float)(RAND_MAX/a);
    if(r2()>=0.5)
        return x;
    else
        return -x;
}

int shuffle_genome_set(genome** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          genome* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

int shuffle_node_set(node** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          node* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

int shuffle_connection_set(connection** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          connection* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

float modified_sigmoid(float x){
    
    return 1/(1+exp(-(4.9)*x));
    
}

genome* init_genome(int input, int output){
    
    int i;
    genome* g = (genome*)malloc(sizeof(genome));
    g->fitness = 0;
    g->specie_rip = 0;
    g->number_input = input;
    g->number_output = output;
    g->number_total_nodes = input+output;
    g->all_nodes = (node**)malloc(sizeof(node*)*(input+output));
    for(i = 0; i < input+output; i++){
        g->all_nodes[i] = (node*)malloc(sizeof(node));
        g->all_nodes[i]->innovation_number = i+1; 
        g->all_nodes[i]->actual_value = 0; 
        g->all_nodes[i]->stored_value = 0;
        g->all_nodes[i]->in_conn_size = 0;
        g->all_nodes[i]->out_conn_size = 0;
        g->all_nodes[i]->in_connections = NULL;
        g->all_nodes[i]->out_connections = NULL;
        
    }
    
    return g;
}

void print_genome(genome* g){
    
    int i,j;
    
    for(i = 0; i < g->number_total_nodes; i++){
        printf("NODE:\n");
        printf("Innovation number: %d\n",g->all_nodes[i]->innovation_number);
        printf("Actual value: %f\n",g->all_nodes[i]->actual_value);
        printf("Stored value: %f\n",g->all_nodes[i]->stored_value);
        printf("In connection size: %d\n",g->all_nodes[i]->in_conn_size);
        printf("out connection size: %d\n",g->all_nodes[i]->out_conn_size);
        printf("IN CONNECTIONS\n");
        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            printf("Innovation number: %d\n",g->all_nodes[i]->in_connections[j]->innovation_number);
            printf("Flag: %d\n",g->all_nodes[i]->in_connections[j]->flag);
            printf("Weight: %f\n",g->all_nodes[i]->in_connections[j]->weight);
            printf("In node: %d\n",g->all_nodes[i]->in_connections[j]->in_node->innovation_number);
            printf("Out node: %d\n",g->all_nodes[i]->in_connections[j]->out_node->innovation_number);
        }
        printf("OUT CONNECTIONS\n");
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            printf("Innovation number: %d\n",g->all_nodes[i]->out_connections[j]->innovation_number);
            printf("Flag: %d\n",g->all_nodes[i]->out_connections[j]->flag);
            printf("Weight: %f\n",g->all_nodes[i]->out_connections[j]->weight);
            printf("In node: %d\n",g->all_nodes[i]->out_connections[j]->in_node->innovation_number);
            printf("Out node: %d\n",g->all_nodes[i]->out_connections[j]->out_node->innovation_number);
        }
    }
    
    
}

void free_genome(genome* g,int global_inn_numb_connections){
    int i,j,counter = 0;
    int* temp = (int*)calloc(global_inn_numb_connections,sizeof(int));
    connection** temp_connection = (connection**)malloc(sizeof(connection*)*global_inn_numb_connections);
    for(i = 0; i < g->number_total_nodes; i++){
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            if(!temp[g->all_nodes[i]->out_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->out_connections[j]->innovation_number-1] = 1;
                temp_connection[counter] = g->all_nodes[i]->out_connections[j];
                counter++;
            }
        }

        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            if(!temp[g->all_nodes[i]->in_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->in_connections[j]->innovation_number-1] = 1;
                temp_connection[counter] = g->all_nodes[i]->in_connections[j];
                counter++;
            }
        }
        
        if(g->all_nodes[i]->in_connections!=NULL)
            free(g->all_nodes[i]->in_connections);
        if(g->all_nodes[i]->out_connections!=NULL)
            free(g->all_nodes[i]->out_connections);
        free(g->all_nodes[i]);
    }
    
    for(i = 0; i < counter; i++){
        free(temp_connection[i]);
    }
    
    free(temp_connection);
    free(temp);
    free(g->all_nodes);
    free(g);
}

genome* copy_genome(genome* g){
    
    int i,j,k,z;
    int* inn_numb_conn = NULL;
    
    genome* new_g = (genome*)malloc(sizeof(genome));
    new_g->fitness = g->fitness;
    new_g->specie_rip = g->specie_rip;
    new_g->number_input = g->number_input;
    new_g->number_output = g->number_output;
    new_g->number_total_nodes = g->number_total_nodes;
    
    new_g->all_nodes = (node**)malloc(sizeof(node*)*new_g->number_total_nodes);
    
    for(i = 0; i < new_g->number_total_nodes; i++){
        new_g->all_nodes[i] = (node*)malloc(sizeof(node));
        new_g->all_nodes[i]->actual_value = g->all_nodes[i]->actual_value;
        new_g->all_nodes[i]->stored_value = g->all_nodes[i]->stored_value;
        new_g->all_nodes[i]->innovation_number = g->all_nodes[i]->innovation_number;
        new_g->all_nodes[i]->in_conn_size = g->all_nodes[i]->in_conn_size;
        new_g->all_nodes[i]->out_conn_size = g->all_nodes[i]->out_conn_size;
        new_g->all_nodes[i]->in_connections = (connection**)malloc(sizeof(connection*)*new_g->all_nodes[i]->in_conn_size);
        new_g->all_nodes[i]->out_connections = (connection**)malloc(sizeof(connection*)*new_g->all_nodes[i]->out_conn_size);
    }
    for(i = 0; i < new_g->number_total_nodes; i++){
        for(j = 0; j < new_g->all_nodes[i]->in_conn_size; j++){
                new_g->all_nodes[i]->in_connections[j] = (connection*)malloc(sizeof(connection));
                new_g->all_nodes[i]->in_connections[j] ->innovation_number = g->all_nodes[i]->in_connections[j]->innovation_number;
                new_g->all_nodes[i]->in_connections[j] ->flag = g->all_nodes[i]->in_connections[j] ->flag;
                new_g->all_nodes[i]->in_connections[j] ->weight = g->all_nodes[i]->in_connections[j] ->weight;
                new_g->all_nodes[i]->in_connections[j]->out_node = new_g->all_nodes[i];
        }
            
    }
        
    for(i = 0; i < new_g->number_total_nodes; i++){
        for(j = 0; j < new_g->all_nodes[i]->out_conn_size; j++){
            for(k = 0; k < new_g->number_total_nodes; k++){
                if(g->all_nodes[i]->out_connections[j]->out_node->innovation_number == new_g->all_nodes[k]->innovation_number){
                    for(z = 0; z < new_g->all_nodes[k]->in_conn_size; z++){
                        if(new_g->all_nodes[k]->in_connections[z]->innovation_number == g->all_nodes[i]->out_connections[j]->innovation_number){
                            new_g->all_nodes[k]->in_connections[z]->in_node = new_g->all_nodes[i];
                            new_g->all_nodes[i]->out_connections[j] = new_g->all_nodes[k]->in_connections[z];
                            break;
                        }
                    }
                    break;
                }
            }
            
        }
        
    }
    
    return new_g;
}

int random_number(int min, int max){
    return (int)((rand() % (max - min)) + min);
}

void init_global_params(int input, int output, int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections){
    int i;
    
    (*global_inn_numb_nodes) = input+output;
    (*global_inn_numb_connections) = 0;
    (*matrix_nodes) = (int**)malloc(sizeof(int*)*(input+output));
    for(i = 0; i < (input+output); i++){
        (*matrix_nodes)[i] = (int*)malloc(sizeof(int)*2);
        (*matrix_nodes)[i][0] = -1;
        (*matrix_nodes)[i][1] = -1;
    }
    (*dict_connections) = NULL;
    (*matrix_connections) = NULL;
}

/*returns a matrix of connection where rows = global_inn_numb_connections*/
connection** get_connections(genome* g, int global_inn_numb_connections){
    int i,j,counter = 0;
    int* temp = (int*)calloc(global_inn_numb_connections,sizeof(int));
    connection** temp_connection = (connection**)malloc(sizeof(connection*)*global_inn_numb_connections);
    for(i = 0; i < g->number_total_nodes; i++){
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            if(!temp[g->all_nodes[i]->out_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->out_connections[j]->innovation_number-1] = 1;
                temp_connection[counter] = g->all_nodes[i]->out_connections[j];
                counter++;
            }
        }

        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            if(!temp[g->all_nodes[i]->in_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->in_connections[j]->innovation_number-1] = 1;
                temp_connection[counter] = g->all_nodes[i]->in_connections[j];
                counter++;
            }
        }
    }
    
    free(temp);
    return temp_connection;
    
}

int get_numb_connections(genome* g, int global_inn_numb_connections){
    int i,j,counter = 0;
    int* temp = (int*)calloc(global_inn_numb_connections,sizeof(int));
    for(i = 0; i < g->number_total_nodes; i++){
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            if(!temp[g->all_nodes[i]->out_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->out_connections[j]->innovation_number-1] = 1;
                counter++;
            }
        }

        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            if(!temp[g->all_nodes[i]->in_connections[j]->innovation_number-1]){
                temp[g->all_nodes[i]->in_connections[j]->innovation_number-1] = 1;
                counter++;
            }
        }
    }
    
    free(temp);
    return counter;
    
}

void adjust_genome(genome* g){
    int i,j;
    int* nodes = (int*)malloc(sizeof(int)*g->number_total_nodes);
    int* nodes_indices = (int*)malloc(sizeof(int)*g->number_total_nodes);
    int* connections;
    int* connections_indices;
    connection** conn;
    for(i = 0; i < g->number_total_nodes; i++){
        nodes[i] = g->all_nodes[i]->innovation_number;
        nodes_indices[i] = i;
    }
    quick_sort_int(nodes,nodes_indices,0,g->number_total_nodes-1);
    node** all_nodes = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    for(i = 0; i < g->number_total_nodes; i++){
        all_nodes[i] = g->all_nodes[nodes_indices[i]];
    }
    free(g->all_nodes);
    g->all_nodes = all_nodes;
    free(nodes);
    free(nodes_indices);
    for(i = 0; i < g->number_total_nodes; i++){
        connections = (int*)malloc(sizeof(int)*g->all_nodes[i]->in_conn_size);
        connections_indices = (int*)malloc(sizeof(int)*g->all_nodes[i]->in_conn_size);
        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            connections[j] = g->all_nodes[i]->in_connections[j]->innovation_number;
            connections_indices[j] = j;
        }
        quick_sort_int(connections,connections_indices,0,g->all_nodes[i]->in_conn_size-1);
        conn = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->in_conn_size);
        for(j = 0; j < g->all_nodes[i]->in_conn_size; j++){
            conn[j] = g->all_nodes[i]->in_connections[connections_indices[j]];
        }
        free(connections);
        free(connections_indices);
        free(g->all_nodes[i]->in_connections);
        g->all_nodes[i]->in_connections = conn;
        
        connections = (int*)malloc(sizeof(int)*g->all_nodes[i]->out_conn_size);
        connections_indices = (int*)malloc(sizeof(int)*g->all_nodes[i]->out_conn_size);
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            connections[j] = g->all_nodes[i]->out_connections[j]->innovation_number;
            connections_indices[j] = j;
        }
        quick_sort_int(connections,connections_indices,0,g->all_nodes[i]->out_conn_size-1);
        conn = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
        for(j = 0; j < g->all_nodes[i]->out_conn_size; j++){
            conn[j] = g->all_nodes[i]->out_connections[connections_indices[j]];
        }
        free(connections);
        free(connections_indices);
        free(g->all_nodes[i]->out_connections);
        g->all_nodes[i]->out_connections = conn;
        
    }
}

