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
    

    fwrite(&g->number_input,sizeof(int),1,write_ptr);
    fwrite(&g->number_output,sizeof(int),1,write_ptr);
    fwrite(&g->number_total_nodes,sizeof(int),1,write_ptr);
    fwrite(&g->fitness,sizeof(float),1,write_ptr);
    
    for(i = 0; i < g->number_total_nodes; i++){
        fwrite(&g->all_nodes[i]->in_conn_size,sizeof(int),1,write_ptr);
        fwrite(&g->all_nodes[i]->out_conn_size,sizeof(int),1,write_ptr);
        fwrite(&g->all_nodes[i]->innovation_number,sizeof(int),1,write_ptr);
    }
    
    fwrite(&n,sizeof(int),1,write_ptr);
    
    
    for(i = 0; i < n; i++){
        fwrite(&c[i]->innovation_number,sizeof(int),1,write_ptr);
        fwrite(&c[i]->in_node->innovation_number,sizeof(int),1,write_ptr);
        fwrite(&c[i]->out_node->innovation_number,sizeof(int),1,write_ptr);
        fwrite(&c[i]->weight,sizeof(float),1,write_ptr);
        fwrite(&c[i]->flag,sizeof(int),1,write_ptr);
    }
    
    free(c);
    i = fclose(write_ptr);
    
    return i;
    
}

genome* load_genome(int global_inn_numb_connections){
    int i,j,n,inn,inn2,k;
    char input[256];
    FILE *read_ptr = NULL;
    connection** c = (connection**)malloc(sizeof(connection*)*global_inn_numb_connections);
    for(i = 0; i < global_inn_numb_connections; i++){
        c[i] = NULL;
    }
    genome* g = (genome*)malloc(sizeof(genome));
    
    do{
        printf("File.bin of the network: ");
        i = scanf("%s",input);
        read_ptr = fopen(input,"r");
    }while(read_ptr == NULL);
    
    k = fread(&g->number_input,sizeof(int),1,read_ptr);
    k = fread(&g->number_output,sizeof(int),1,read_ptr);
    k = fread(&g->number_total_nodes,sizeof(int),1,read_ptr);
    k = fread(&g->fitness,sizeof(float),1,read_ptr);
    g->all_nodes = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    
    for(i = 0; i < g->number_total_nodes; i++){
        g->all_nodes[i] = (node*)malloc(sizeof(node));
        k = fread(&g->all_nodes[i]->in_conn_size,sizeof(int),1,read_ptr);
        k = fread(&g->all_nodes[i]->out_conn_size,sizeof(int),1,read_ptr);
        k = fread(&g->all_nodes[i]->innovation_number,sizeof(int),1,read_ptr);
        g->all_nodes[i]->actual_value = 0;
        g->all_nodes[i]->stored_value = 0;
        g->all_nodes[i]->in_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->in_conn_size);
        g->all_nodes[i]->out_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
    }
    
    k = fread(&n,sizeof(int),1,read_ptr);

    for(i = 0; i < n; i++){
        k = fread(&inn,sizeof(int),1,read_ptr);
        free(c[inn-1]);
        c[inn-1] = (connection*)malloc(sizeof(connection));
        c[inn-1]->innovation_number = inn;
        k = fread(&inn2,sizeof(int),1,read_ptr);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2)
                c[inn-1]->in_node = g->all_nodes[j];
        }
        k = fread(&inn2,sizeof(int),1,read_ptr);
        for(j = 0; j < g->number_total_nodes; j++){
            if(g->all_nodes[j]->innovation_number == inn2)
                c[inn-1]->out_node = g->all_nodes[j];
        }
        
        k = fread(&c[inn-1]->weight,sizeof(float),1,read_ptr);
        k = fread(&c[inn-1]->flag,sizeof(int),1,read_ptr);
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

