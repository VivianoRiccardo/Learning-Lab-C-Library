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

//dict connections: dato una connessione con i = innovation number-1 della connessione allora dict_connections[i] = 0 se questa
//connessione non è mai stata splittata globalmente, altrimenti restituisce l'innovation number del neurone generato da questo
//splittaggio
//matrix nodes: è grande quanto il numero globale di innovation number per i nodi
//i = innovation number-1 di un nodo splittato, matrix_nodes[i][0] è l'innovation number della nuova connessione di input del nodo splittato
//matrix_nodes[i][1] è l'innovation number della nuova connessione di output del nodo splittato
//matrix connections: i: = innovation number -1 di una connessione
//matrix_connections[i][0] è l'innovation number del neurone di input di quella connessione
//matrix_connections[i][1] è l'innovation number del neurone di output di quella connessione
//per chiamare split random connection bisogna avere almeno una connessione che non è mai stata splittata altrimenti si incastona nel while

void connections_mutation(genome* g, int global_inn_numb_connections, float first_thereshold, float second_thereshold){
    connection** c = get_connections(g,global_inn_numb_connections);
    int i,n = get_numb_connections(g,global_inn_numb_connections);
    
    for(i = 0; i < n; i++){
        if(r2() < first_thereshold){
            if(r2()<second_thereshold)
                c[i]->weight = random_float_number(5);
            else
                c[i]->weight += random_normal();
            
        }
    }
    
    free(c);
}
int split_random_connection(genome* g,int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections){
    
    int i,j,k,flag1 = 0,flag3 = 0,flag2 = 0,c1 = 0,c2 = 0;
    int** new_matrix_nodes = NULL;
    int** new_matrix_connections = NULL;
    int* new_dict_connections = NULL;
    connection** new_out_connections = NULL;
    connection** new_in_connections = NULL;
    node** new_all_nodes = NULL;
    
    node** temp_node = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    for(i = 0; i < g->number_total_nodes; i++){
        temp_node[i] = g->all_nodes[i];
    }
    
    shuffle_node_set(temp_node,g->number_total_nodes);
    for(i = 0; i < g->number_total_nodes; i++){
        flag1 = 0;
        flag2 = 1;
        flag3= 1;
        //se il neurone random non ha connessioni di output da splittare si ricicla il while
        if(temp_node[i]->out_conn_size == 0)
            flag1 = 1;
        if(!flag1){
            for(j = 0; j < temp_node[i]->out_conn_size; j++){
                flag2 = 0;
                

                //se è stata scelta una connessione random disattivata di un neurone random si ricicla il while

                if(!temp_node[i]->out_connections[j]->flag)
                    flag2 = 1;
                if(!flag2){
                    flag3 = 0;
                    if((*dict_connections)[temp_node[i]->out_connections[j]->innovation_number-1]){
                        for(k = 0; k < g->number_total_nodes; k++){
                            //se la connessione per questo genoma è stata già splittata in passato e quindi esiste il neurone che si genera
                            //da questo splittaggio allora si ricila il while
                            if(g->all_nodes[k]->innovation_number == (*dict_connections)[temp_node[i]->out_connections[j]->innovation_number-1]){
                                flag3 = 1;
                                break;
                            }
                        }
                    }
                }
                if(!flag2 && !flag3)
                    break;
            }
            
        }
        if(!flag1 && !flag2 && !flag3)
            break;
    }
    
    
    if(!flag1 && !flag2 && !flag3){
        for(k = 0; k < g->number_total_nodes; k++){
            if(g->all_nodes[k]->innovation_number == temp_node[i]->innovation_number){
                break;
            }
        }
        i = k;
    }
    
    else{
        free(temp_node);
        return 0;
    }
    
    free(temp_node);
    
    if(!(*dict_connections)[g->all_nodes[i]->out_connections[j]->innovation_number-1]){
        /*global*/
        (*dict_connections)[g->all_nodes[i]->out_connections[j]->innovation_number-1] = (*global_inn_numb_nodes)+1;
        (*global_inn_numb_nodes)++;
        (*global_inn_numb_connections)+=2;
        
        new_matrix_nodes = (int**)malloc(sizeof(int*)*(*global_inn_numb_nodes));
        
        for(k = 0; k < (*global_inn_numb_nodes) -1; k++){
            new_matrix_nodes[k] = (*matrix_nodes)[k];
        }
        
        new_matrix_nodes[k] = (int*)malloc(sizeof(int)*2);
        new_matrix_nodes[k][0] = (*global_inn_numb_connections) -1;
        new_matrix_nodes[k][1] = (*global_inn_numb_connections);
        
        if((*matrix_nodes!=NULL))
            free((*matrix_nodes));
            
        (*matrix_nodes) = new_matrix_nodes;
        
        new_matrix_connections = (int**)malloc(sizeof(int*)*(*global_inn_numb_connections));
        
        for(k = 0; k < (*global_inn_numb_connections)-2; k++){
            new_matrix_connections[k] = (*matrix_connections)[k];
        }
        
        new_matrix_connections[k] = (int*)malloc(sizeof(int)*2);
        new_matrix_connections[k][0] = g->all_nodes[i]->innovation_number;
        new_matrix_connections[k][1] = (*global_inn_numb_nodes);
        
        new_matrix_connections[k+1] = (int*)malloc(sizeof(int)*2);
        new_matrix_connections[k+1][0] = (*global_inn_numb_nodes);
        new_matrix_connections[k+1][1] = g->all_nodes[i]->out_connections[j]->out_node->innovation_number;
        
        if((*matrix_connections)!=NULL)
            free((*matrix_connections));

        (*matrix_connections) = new_matrix_connections;
        
        
        new_dict_connections = (int*)malloc(sizeof(int)*(*global_inn_numb_connections));
        
        for(k = 0; k < (*global_inn_numb_connections)-2; k++){
            new_dict_connections[k] = (*dict_connections)[k];
        }
        
        new_dict_connections[k] = 0;
        new_dict_connections[k+1] = 0;
        
        if((*dict_connections)!=NULL)
            free((*dict_connections));
            
        (*dict_connections) = new_dict_connections;
        
    }
    
    /*local*/
    
    g->all_nodes[i]->out_connections[j]->flag = 0;
    g->all_nodes[i]->out_conn_size++;
    g->all_nodes[i]->out_connections[j]->out_node->in_conn_size++;
    
    new_out_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
    for(k = 0; k < g->all_nodes[i]->out_conn_size-1; k++){
        new_out_connections[k] = g->all_nodes[i]->out_connections[k];
    }
    
    new_out_connections[k] = (connection*)malloc(sizeof(connection));
    if(g->all_nodes[i]->out_connections!=NULL)
        free(g->all_nodes[i]->out_connections);
    
    g->all_nodes[i]->out_connections = new_out_connections;
    
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->innovation_number = (*matrix_nodes)[(*dict_connections)[g->all_nodes[i]->out_connections[j]->innovation_number-1]-1][0];
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->weight = 1;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->flag = 1;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->in_node = g->all_nodes[i];
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node = (node*)malloc(sizeof(node));
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->innovation_number = (*dict_connections)[g->all_nodes[i]->out_connections[j]->innovation_number-1];
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->actual_value = 0;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->stored_value = 0;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->in_conn_size = 1;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_conn_size = 1;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->in_connections = (connection**)malloc(sizeof(connection*));
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections = (connection**)malloc(sizeof(connection*));
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->in_connections[0] = g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1];
    
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0] = (connection*)malloc(sizeof(connection));
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0]->innovation_number = (*matrix_nodes)[(*dict_connections)[g->all_nodes[i]->out_connections[j]->innovation_number-1]-1][1];
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0]->weight = g->all_nodes[i]->out_connections[j]->weight;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0]->flag = 1;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0]->in_node = g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node;
    g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0]->out_node = g->all_nodes[i]->out_connections[j]->out_node;
    
    new_in_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_connections[j]->out_node->in_conn_size);
    
    for(k = 0; k < g->all_nodes[i]->out_connections[j]->out_node->in_conn_size-1; k++){
        new_in_connections[k] = g->all_nodes[i]->out_connections[j]->out_node->in_connections[k];
    }
    
    new_in_connections[k] = g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node->out_connections[0];
    if(g->all_nodes[i]->out_connections[j]->out_node->in_connections!=NULL)
        free(g->all_nodes[i]->out_connections[j]->out_node->in_connections);
    
    g->all_nodes[i]->out_connections[j]->out_node->in_connections = new_in_connections;
    
    g->number_total_nodes++;
    
    new_all_nodes = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    for(k = 0; k < g->number_total_nodes-1; k++){
        new_all_nodes[k] = g->all_nodes[k];
    }
    
    new_all_nodes[k] = g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1]->out_node;
    if(g->all_nodes!=NULL)
        free(g->all_nodes);
    
    g->all_nodes = new_all_nodes;
    
    return 1;
    
}

/*there must be at least 2 nodes not connected in a specific direction*/
int add_random_connection(genome* g,int* global_inn_numb_connections, int*** matrix_connections, int** dict_connections){
    int i,j,k,z,flag = 0,count1 = 0, count2= 0,n_connections;
    int** new_matrix_connections = NULL;
    int* new_dict_connections = NULL;
    connection** new_out_connections = NULL;
    connection** new_in_connections = NULL;
    connection** temp_connections = NULL;
    
    node** temp_node1 = (node**)malloc(sizeof(node*)*(g->number_total_nodes-g->number_output));
    node** temp_node2 = (node**)malloc(sizeof(node*)*(g->number_total_nodes-g->number_input));
    for(i = 0; i < g->number_total_nodes; i++){
        if(g->all_nodes[i]->innovation_number-1 < g->number_input){
            temp_node1[count1] = g->all_nodes[i];
            count1++;
        }
        
        else if(g->all_nodes[i]->innovation_number-1 >= g->number_input && g->all_nodes[i]->innovation_number-1 < g->number_input+g->number_output){
            temp_node2[count2] = g->all_nodes[i];
            count2++;
        }
        
        else{
            temp_node1[count1] = g->all_nodes[i];
            temp_node2[count2] = g->all_nodes[i];
            count1++;
            count2++;
        }
    }
    
    shuffle_node_set(temp_node1,count1);
    shuffle_node_set(temp_node2,count2);
    
    //si scelgono 2 neuroni random il primo [i] fa da input il secondo [j] fa da output per la nuova connessione
    //se non c'è già il collegamento neurone di input->neurone di output allora si esce dal while
    //altrimenti si ricicla, edit: se la connessione c'è già ma è disattivata viene attivata e si esce
    n_connections = get_numb_connections(g,(*global_inn_numb_connections));
    temp_connections = get_connections(g,(*global_inn_numb_connections));
    
    for(i = 0; i < count1; i++){
        flag = 0;
        for(j = 0; j < count2; j++){
            flag = 0;
            for(k = 0; k < n_connections; k++){
                if(temp_connections[k]->in_node->innovation_number == temp_node1[i]->innovation_number && temp_connections[k]->out_node->innovation_number == temp_node2[j]->innovation_number){
                    if(!temp_connections[k]->flag){
                        temp_connections[k]->flag = 1;
                        free(temp_connections);
                        free(temp_node1);
                        free(temp_node2);
                        return 1;
                    }
                    
                    else{
                        flag = 1;
                        break;
                    }
                    
                } 
            }
            if(!flag)
                break;
        }
        if(!flag)
            break;
    }
    
    if(flag){
        free(temp_connections);
        free(temp_node1);
        free(temp_node2);
        return 0;
    }
    
    else{
        for(k = 0; k < g->number_total_nodes; k++){
            if(g->all_nodes[k]->innovation_number == temp_node1[i]->innovation_number)
                break;
        }
        i = k;
        for(k = 0; k < g->number_total_nodes; k++){
            if(g->all_nodes[k]->innovation_number == temp_node2[j]->innovation_number)
                break;
        }
        j = k;
    }
    
    free(temp_connections);
    free(temp_node1);
    free(temp_node2);
    
        
    flag = 0;
    
    for(k = 0; k < (*global_inn_numb_connections); k++){
        if((*matrix_connections)[k][0] == g->all_nodes[i]->innovation_number && (*matrix_connections)[k][1] == g->all_nodes[j]->innovation_number){
            flag = 1;
            z = k;
            break;
        }
    }
    
    if(!flag){
        /*global*/
        (*global_inn_numb_connections)++;
        
        new_dict_connections = (int*)malloc(sizeof(int)*(*global_inn_numb_connections));
        for(k = 0; k < (*global_inn_numb_connections)-1; k++){
            new_dict_connections[k] = (*dict_connections)[k];
        }
        
        new_dict_connections[k] = 0;
        if((*dict_connections)!=NULL)
            free((*dict_connections));
            
        (*dict_connections) = new_dict_connections;
        
        
        new_matrix_connections = (int**)malloc(sizeof(int*)*(*global_inn_numb_connections));
        
        for(k = 0; k < (*global_inn_numb_connections)-1; k++){
            new_matrix_connections[k] = (*matrix_connections)[k];
        }
        
        new_matrix_connections[k] = (int*)malloc(sizeof(int)*2);
        new_matrix_connections[k][0] = g->all_nodes[i]->innovation_number;
        new_matrix_connections[k][1] = g->all_nodes[j]->innovation_number;
        
        if((*matrix_connections)!=NULL)
            free((*matrix_connections));
        
        (*matrix_connections) = new_matrix_connections;
        z = (*global_inn_numb_connections)-1;
        
    }
    
    /*local*/
    g->all_nodes[i]->out_conn_size++;
    g->all_nodes[j]->in_conn_size++;
    new_out_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[i]->out_conn_size);
    for(k = 0; k < g->all_nodes[i]->out_conn_size-1; k++){
        new_out_connections[k] = g->all_nodes[i]->out_connections[k];
    }
    
    new_out_connections[k] = (connection*)malloc(sizeof(connection));
    new_out_connections[k]->innovation_number = z+1;
    new_out_connections[k]->flag = 1;
    new_out_connections[k]->weight = random_float_number(2);
    new_out_connections[k]->in_node = g->all_nodes[i];
    new_out_connections[k]->out_node = g->all_nodes[j];
    
    if(g->all_nodes[i]->out_connections!=NULL)
        free(g->all_nodes[i]->out_connections);
    g->all_nodes[i]->out_connections = new_out_connections;
    
    new_in_connections = (connection**)malloc(sizeof(connection*)*g->all_nodes[j]->in_conn_size);
    
    for(k = 0; k < g->all_nodes[j]->in_conn_size-1; k++){
        new_in_connections[k] = g->all_nodes[j]->in_connections[k];
    }
    
    new_in_connections[k] = g->all_nodes[i]->out_connections[g->all_nodes[i]->out_conn_size-1];
    if(g->all_nodes[j]->in_connections!=NULL)
        free(g->all_nodes[j]->in_connections);
    
    g->all_nodes[j]->in_connections = new_in_connections;
    
    return 1;
}

int remove_random_connection(genome* g, int global_inn_numb_connections){
    connection** c = get_connections(g,global_inn_numb_connections);
    int i,flag = 0,n = get_numb_connections(g,global_inn_numb_connections);
    
    if(!n){
        free(c);
        return 0;
    }
    
    shuffle_connection_set(c,n);
    
    for(i = 0; i < n; i++){
        if(c[i]->flag){
            c[i]->flag = 0;
            flag = 1;
            break;
        }
    }
    
    free(c);
    
    if(!flag)
        return 0;
    else
        return 1;
    
}

int activate_random_connection(genome* g, int global_inn_numb_connections){
    connection** c = get_connections(g,global_inn_numb_connections);
    int i,flag = 0,n = get_numb_connections(g,global_inn_numb_connections);
    
    if(!n){
        free(c);
        return 0;
    }
    
    shuffle_connection_set(c,n);
    
    for(i = 0; i < n; i++){
        if(!c[i]->flag){
            c[i]->flag = 1;
            flag = 1;
            break;
        }
    }
    
    free(c);
    
    if(!flag)
        return 0;
    else
        return 1;
    
}

int activate_connections(genome* g, int global_inn_numb_connections,float thereshold){
    connection** c = get_connections(g,global_inn_numb_connections);
    int i,flag = 0,n = get_numb_connections(g,global_inn_numb_connections);
    
    if(!n){
        free(c);
        return 0;
    }
    
    shuffle_connection_set(c,n);
    
    for(i = 0; i < n; i++){
        if(!c[i]->flag){
            if(r2()<thereshold){
                c[i]->flag = 1;
                flag = 1;
            }
        }
    }
    
    free(c);
    
    if(!flag)
        return 0;
    else
        return 1;
    
}

genome* crossover(genome* g, genome* g2, int global_inn_numb_connections,int global_inn_numb_nodes){
    
    genome* g1 = copy_genome(g);
    int i,j,n1,n2,flag,count_n = 0,count_c = 0,k;
    int* node_array = (int*)calloc(global_inn_numb_nodes,sizeof(int));
    connection** c1;
    connection** c2;
    node** temp_node = NULL;
    connection** temp_connection = NULL;
    node** new_total_nodes = NULL;
    connection** t_c = NULL;
    node** t_n = NULL;
    
    n1 = get_numb_connections(g1,global_inn_numb_connections);
    c1 = get_connections(g1,global_inn_numb_connections);
    
    n2 = get_numb_connections(g2,global_inn_numb_connections);
    c2 = get_connections(g2,global_inn_numb_connections);
    
    for(i = 0; i < g2->number_total_nodes; i++){
        flag = 0;
        for(j = 0; j < g1->number_total_nodes; j++){
            if(g1->all_nodes[j]->innovation_number == g2->all_nodes[i]->innovation_number){
                flag = 1;
                break;
            }
        }
        if(!flag){
            node_array[g2->all_nodes[i]->innovation_number-1] = 1;
            if(!count_n)
                temp_node = (node**)malloc(sizeof(node*));
            else{
                t_n = (node**)malloc(sizeof(node*)*(count_n+1));
                for(k = 0; k < count_n; k++){
                    t_n[k] = temp_node[k];
                }                
            }
            
            if(!count_n){    
                temp_node[count_n] = (node*)malloc(sizeof(node));
                temp_node[count_n]->innovation_number = g2->all_nodes[i]->innovation_number;
                temp_node[count_n]->actual_value = g2->all_nodes[i]->actual_value;
                temp_node[count_n]->stored_value = g2->all_nodes[i]->stored_value;
                temp_node[count_n]->in_conn_size = 0;
                temp_node[count_n]->out_conn_size = 0;
                temp_node[count_n]->in_connections = NULL;
                temp_node[count_n]->out_connections = NULL;
            }
            
            else{
                t_n[count_n] = (node*)malloc(sizeof(node));
                t_n[count_n]->innovation_number = g2->all_nodes[i]->innovation_number;
                t_n[count_n]->actual_value = g2->all_nodes[i]->actual_value;
                t_n[count_n]->stored_value = g2->all_nodes[i]->stored_value;
                t_n[count_n]->in_conn_size = 0;
                t_n[count_n]->out_conn_size = 0;
                t_n[count_n]->in_connections = NULL;
                t_n[count_n]->out_connections = NULL;
                free(temp_node);
                temp_node = t_n;
            }
            count_n++;
            
        }
    }
    
    for(i = 0; i < n2; i++){
        flag = 0;
        for(j = 0; j < n1; j++){
            if(c1[j]->innovation_number == c2[i]->innovation_number){
                flag = 1;
                break;
            }
        }
        if(!flag){
                        
            if(!count_c)
                temp_connection = (connection**)malloc(sizeof(connection*));
            else{
                t_c = (connection**)malloc(sizeof(connection*)*(count_c+1));
                for(k = 0; k < count_c; k++){
                    t_c[k] = temp_connection[k];
                }
                
            }
            
            if(!count_c){
                temp_connection[count_c] = (connection*)malloc(sizeof(connection));
                temp_connection[count_c]->innovation_number = c2[i]->innovation_number;
                temp_connection[count_c]->weight = c2[i]->weight;
                temp_connection[count_c]->flag = c2[i]->flag;
            }
            
            else{
                t_c[count_c] = (connection*)malloc(sizeof(connection));
                t_c[count_c]->innovation_number = c2[i]->innovation_number;
                t_c[count_c]->weight = c2[i]->weight;
                t_c[count_c]->flag = c2[i]->flag;
                free(temp_connection);
                temp_connection = t_c;                
            }
            
            if(node_array[c2[i]->in_node->innovation_number-1]){
                for(j = 0; j < count_n; j++){
                    if(temp_node[j]->innovation_number == c2[i]->in_node->innovation_number){
                        if(!temp_node[j]->out_conn_size)
                            temp_node[j]->out_connections = (connection**)malloc(sizeof(connection*));
                        else{
                            t_c = (connection**)malloc(sizeof(connection*)*(temp_node[j]->out_conn_size+1));
                            for(k = 0; k < temp_node[j]->out_conn_size; k++){
                                t_c[k] = temp_node[j]->out_connections[k];
                            }
                        }
                        if(!temp_node[j]->out_conn_size)
                            temp_node[j]->out_connections[temp_node[j]->out_conn_size] = temp_connection[count_c];
                            
                        else{
                            t_c[temp_node[j]->out_conn_size] = temp_connection[count_c];                            
                            free(temp_node[j]->out_connections);
                            temp_node[j]->out_connections = t_c;
                        }
                        temp_node[j]->out_conn_size++;
                        temp_connection[count_c]->in_node = temp_node[j];
                        break;    
                    }
                }
            }
            
            else{
                for(j = 0; j < g1->number_total_nodes; j++){
                    if(g1->all_nodes[j]->innovation_number == c2[i]->in_node->innovation_number){
                        if(!g1->all_nodes[j]->out_conn_size){
                            free(g1->all_nodes[j]->out_connections);
                            g1->all_nodes[j]->out_connections = (connection**)malloc(sizeof(connection*));
                            
                        }
                        else{
                            t_c = (connection**)malloc(sizeof(connection*)*(g1->all_nodes[j]->out_conn_size+1));
                            for(k = 0; k < g1->all_nodes[j]->out_conn_size; k++){
                                t_c[k] = g1->all_nodes[j]->out_connections[k];
                            }
                        }
                        if(!g1->all_nodes[j]->out_conn_size)
                            g1->all_nodes[j]->out_connections[g1->all_nodes[j]->out_conn_size] = temp_connection[count_c];
                            
                        else{
                            t_c[g1->all_nodes[j]->out_conn_size] = temp_connection[count_c];                            
                            free(g1->all_nodes[j]->out_connections);
                            g1->all_nodes[j]->out_connections = t_c;
                        }
                        g1->all_nodes[j]->out_conn_size++;
                        temp_connection[count_c]->in_node = g1->all_nodes[j];
                        break;    
                    }
                }
            }
            
            if(node_array[c2[i]->out_node->innovation_number-1]){
                for(j = 0; j < count_n; j++){
                    if(temp_node[j]->innovation_number == c2[i]->out_node->innovation_number){
                        if(!temp_node[j]->in_conn_size)
                            temp_node[j]->in_connections = (connection**)malloc(sizeof(connection*));
                        else{
                            t_c = (connection**)malloc(sizeof(connection*)*(temp_node[j]->in_conn_size+1));
                            for(k = 0; k < temp_node[j]->in_conn_size; k++){
                                t_c[k] = temp_node[j]->in_connections[k];
                            }
                        }
                        if(!temp_node[j]->in_conn_size)
                            temp_node[j]->in_connections[temp_node[j]->in_conn_size] = temp_connection[count_c];
                            
                        else{
                            t_c[temp_node[j]->in_conn_size] = temp_connection[count_c];                            
                            free(temp_node[j]->in_connections);
                            temp_node[j]->in_connections = t_c;
                        }
                        temp_node[j]->in_conn_size++;
                        temp_connection[count_c]->out_node = temp_node[j];
                        break;    
                    }
                }
            }
            
            else{
                for(j = 0; j < g1->number_total_nodes; j++){
                    if(g1->all_nodes[j]->innovation_number == c2[i]->out_node->innovation_number){
                        if(!g1->all_nodes[j]->in_conn_size){
                            free(g1->all_nodes[j]->in_connections);
                            g1->all_nodes[j]->in_connections = (connection**)malloc(sizeof(connection*));
                        }
                        else{
                            t_c = (connection**)malloc(sizeof(connection*)*(g1->all_nodes[j]->in_conn_size+1));
                            for(k = 0; k < g1->all_nodes[j]->in_conn_size; k++){
                                t_c[k] = g1->all_nodes[j]->in_connections[k];
                            }
                        }
                        if(!g1->all_nodes[j]->in_conn_size)
                            g1->all_nodes[j]->in_connections[g1->all_nodes[j]->in_conn_size] = temp_connection[count_c];
                            
                        else{
                            t_c[g1->all_nodes[j]->in_conn_size] = temp_connection[count_c];                            
                            free(g1->all_nodes[j]->in_connections);
                            g1->all_nodes[j]->in_connections = t_c;
                        }
                        g1->all_nodes[j]->in_conn_size++;
                        temp_connection[count_c]->out_node = g1->all_nodes[j];
                        break;    
                    }
                }
            }
            count_c++;
            
        }
    }
    
    if(count_n){
        g1->number_total_nodes+=count_n;
        new_total_nodes = (node**)malloc(sizeof(node*)*g1->number_total_nodes);
        for(i = 0; i < g1->number_total_nodes; i++){
            if(i < g1->number_total_nodes-count_n)
                new_total_nodes[i] = g1->all_nodes[i];
            else
                new_total_nodes[i] = temp_node[i-(g1->number_total_nodes-count_n)];
        }
        
        free(g1->all_nodes);
        g1->all_nodes = new_total_nodes;
    }
    
    else
        free(new_total_nodes);

    for(i = 0; i < n1; i++){
        
        for(j = 0; j < n2; j++){
            if(c2[j]->innovation_number == c1[i]->innovation_number){
                if(r2()>=0.5){
                    c1[i]->weight = c2[j]->weight;
                    c1[i]->flag = c2[j]->flag;
                    break;
                }
            }
            
        }
        
    }
    
    
    free(node_array);
    free(c1);
    free(c2);
    
    free(temp_node);
    free(temp_connection);
    
    return g1;
}

void activate_bias(genome* g){
    node** n = (node**)malloc(sizeof(node*)*g->number_total_nodes);
    int i;
    
    for(i = 0; i < g->number_total_nodes; i++){
        n[i] = g->all_nodes[i];
    }
    
    shuffle_node_set(n,g->number_total_nodes);
    for(i = 0; i < g->number_total_nodes; i++){
        if(!n[i]->bias_flag){
            n[i]->bias_flag = 1;
            break;
        }
    }
    free(n);
}
