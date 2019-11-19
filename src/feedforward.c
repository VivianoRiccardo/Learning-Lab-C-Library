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

float* feed_forward(genome* g1, float* inputs, int global_inn_numb_nodes, int global_inn_numb_connections){
    genome* g = copy_genome(g1);
    int i,j,k1,k2,k3,k4,size,global_j,number_connections,flag,there_is_storing = 0;
    int* array = (int*)calloc(global_inn_numb_nodes,sizeof(int));
    int* array2 = (int*)calloc(global_inn_numb_nodes,sizeof(int));
    int** array3 = (int**)malloc(global_inn_numb_nodes*sizeof(int*));
    ff** lists = (ff**)malloc(sizeof(ff*)*g->number_output);
    int* temp = (int*)malloc(sizeof(int)*g->number_output);
    connection** c = NULL;
    float* outputs = (float*)malloc(sizeof(float)*g->number_output);
    
    for(i = 0; i < global_inn_numb_nodes; i++){
        array3[i] = (int*)calloc(global_inn_numb_nodes,sizeof(int));
    }
    for(i = 0; i < g->number_output; i++){
        size = 0;
        global_j = 0;
        ff_reconstruction(g,&array,g->all_nodes[i+g->number_input],1,&lists[i], &size,&global_j);
        temp[i] = size;
    }
    
    
    /*array[i] è uguale ad 1 se il nodo con innovation number i+1 è un input
     * o uno stored node, ovvero un punto di loop*/
    size = 0;
    there_is_storing = 0;
    for(i = 0; i < g->number_output; i++){
        for(j = 0; j < temp[i]; j++){
            array[lists[i][j].list_nodes[lists[i][j].size-1]->innovation_number-1] = lists[i][j].flag+1;
            if(lists[i][j].flag == 1){
                array[lists[i][j].list_nodes[lists[i][j].size-1]->innovation_number-1] = 1;
                there_is_storing = 1;
            }
            
            else if(lists[i][j].flag == -1)
                lists[i][j].list_nodes[lists[i][j].size-1]->flag = 1;

        }
    }
    
    
    /*adesso contrassegniamo le connessioni che sono utilizzate dal nostro nuovo grafo con flag = -1*/
    for(i = 0; i < g->number_output; i++){
        for(j = 0; j < temp[i]; j++){
            
            for(k1 = 0; k1 < lists[i][j].size-1; k1++){
                lists[i][j].list_connections[k1]->flag = -1;
                }
            }
            
        }
    
    
    /*ci sono delle connessioni di alcuni stored node il cui flag va messo a -2, sono quelle
     * connessioni in cui l'in node è uno stored node e la cui sequenza
     * in node out node uno di seguito all'altra non appare in nessun'altra lista nella quale
     * l'ultimo elemento non sia l'in node della connessione*/
    for(i = 0; i < g->number_output; i++){
        for(j = 0; j < temp[i]; j++){
            if(lists[i][j].flag == 1){
                flag = 0;
                for(k1 = 0; k1 < g->number_output; k1++){
                    for(k2 = 0; k2 < temp[k1]; k2++){
                        if(!(k1==i && k2== j)){
                            for(k3 = 0; k3 < lists[k1][k2].size-1; k3++){
                                if(lists[k1][k2].list_nodes[k3]->innovation_number == lists[i][j].list_nodes[lists[i][j].size-2]->innovation_number){
                                    if(lists[k1][k2].list_nodes[k3+1]->innovation_number == lists[i][j].list_nodes[lists[i][j].size-1]->innovation_number && (k3+1 != lists[k1][k2].size-1)){
                                        flag = 1;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if(flag)
                            break;
                    }
                    if(flag)
                        break;
                }
                
                if(!flag)
                    lists[i][j].list_connections[lists[i][j].size-2]->flag = -2;
                
            }
        }
        
    }
    
     
    /*eseguiamo il vero feed forward solo per i veri input e i bias*/
    for(i = 0; i < g->number_input; i++){
        g->all_nodes[i]->actual_value = inputs[i];
    }
    for(i = 0; i < g->number_input; i++){
        if(array[g->all_nodes[i]->innovation_number-1])
            recursive_computation(&array, g->all_nodes[i], g, NULL,&g->all_nodes[i]->actual_value);
    }
    for(i = 0; i < g->number_output; i++){
        for(j = 0; j < temp[i]; j++){
            if(lists[i][j].flag == -1){
                recursive_computation(&array, lists[i][j].list_nodes[lists[i][j].size-1], g, NULL,&lists[i][j].list_nodes[lists[i][j].size-1]->actual_value);
            }
        }
    }
    
    if(there_is_storing){
        
        /*qui i nodi poco dopo er gli stored vanno segnalati*/
        for(i = 0; i < g->number_output; i++){
            for(j = 0; j < temp[i]; j++){
                if(lists[i][j].flag == 1){
                    array2[lists[i][j].list_nodes[lists[i][j].size-2]->innovation_number-1] = 1;
                }
            }
        }
        
        /*si azzerano tutti gli actual value e gli actual value dei nodi di store vengono spostati
         * nello stored value*/
        for(i = 0; i < g->number_total_nodes; i++){
            g->all_nodes[i]->flag = 0;
            if(!array[g->all_nodes[i]->innovation_number-1])
                g->all_nodes[i]->actual_value = 0;
            else{
                if(i>=g->number_input)
                    g->all_nodes[i]->stored_value = g->all_nodes[i]->actual_value;
                g->all_nodes[i]->actual_value = 0;
            }
        }
        
        /*l'actual value degli elementi dopo gli stored node vengono riempiti con la sigmoide per la connessione
         * dello stored value del nodo di stored che li precedono*/
        for(i = 0; i < g->number_output; i++){
            for(j = 0; j < temp[i]; j++){
                if(lists[i][j].flag == 1){
                    if(!array3[lists[i][j].list_nodes[lists[i][j].size-2]->innovation_number-1][lists[i][j].list_nodes[lists[i][j].size-1]->innovation_number-1]){
                        lists[i][j].list_nodes[lists[i][j].size-2]->actual_value = (modified_sigmoid(lists[i][j].list_nodes[lists[i][j].size-1]->stored_value)*(lists[i][j].list_connections[lists[i][j].size-2]->weight));
                        lists[i][j].list_nodes[lists[i][j].size-2]->flag = 1;
                        array3[lists[i][j].list_nodes[lists[i][j].size-2]->innovation_number-1][lists[i][j].list_nodes[lists[i][j].size-1]->innovation_number-1] = 1;
                    }
                }
            }
        }
        
        /*avendo eseguito già un primo feed forward le connessioni che si dovrebbero utilizzare
         * sono state piazzate tutte a -2 dalla ff computation function
         * ora tutte le connessioni di quelle sequenze degli stored vanno attivate (flag = -1) dal nodo
         * subito dopo il nodo di stored*/
        for(i = 0; i < g->number_output; i++){
            for(j = 0; j < temp[i]; j++){
                if(lists[i][j].flag == 1){
                    for(k1 = 0; k1 < lists[i][j].size-2; k1++){
                        lists[i][j].list_connections[k1]->flag = -1;
                    }
                }
                
                else if(lists[i][j].flag == 0 || lists[i][j].flag == -1){
                    lists[i][j].list_nodes[lists[i][j].size-1]->flag = 1;
                    flag = 0;
                    for(k1 = 0; k1 < lists[i][j].size; k1++){
                        if(array2[lists[i][j].list_nodes[k1]->innovation_number-1]){
                            flag = 1;
                            break;    
                        }
                    }
                    
                    if(!flag){
                        for(k1 = 0; k1 < lists[i][j].size-1; k1++){
                            lists[i][j].list_connections[k1]->flag = -1;
                        }
                    }
                }
                
            }
        }
        
        for(i = 0; i < g->number_input; i++){
            g->all_nodes[i]->actual_value = inputs[i];
        }
        
        for(i = 0; i < g->number_output; i++){
            for(j = 0; j < temp[i]; j++){
                if(lists[i][j].flag == 1){
                    recursive_computation(&array, lists[i][j].list_nodes[lists[i][j].size-2], g, NULL,&lists[i][j].list_nodes[lists[i][j].size-2]->actual_value);
                }
            }
        }
        
        for(i = 0; i < g->number_input; i++){
            if(array[g->all_nodes[i]->innovation_number-1])
                recursive_computation(&array, g->all_nodes[i], g, NULL,&g->all_nodes[i]->actual_value);
        }
        
        for(i = 0; i < g->number_output; i++){
            for(j = 0; j < temp[i]; j++){
                if(lists[i][j].flag == -1){
                    recursive_computation(&array, lists[i][j].list_nodes[lists[i][j].size-1], g, NULL,&lists[i][j].list_nodes[lists[i][j].size-1]->actual_value);
                }
            }
        }
        
    }
    
    for(i = 0; i < g->number_output; i++){
        outputs[i] = modified_sigmoid(g->all_nodes[i+g->number_input]->actual_value+1);
    }
    
    /*deallocation*/
    for(i = 0; i < g->number_output; i++){
        for(j = 0; j < temp[i]; j++){
            free(lists[i][j].list_nodes);
            
            free(lists[i][j].list_connections);
        }
        
        
        free(lists[i]);
    }
    
    for(i = 0; i < global_inn_numb_nodes; i++){
        free(array3[i]);
    }

    free(array3);
    free(array2);
    free(temp);
    free(lists);
    free(array);
    free(c);
    free_genome(g,global_inn_numb_connections);
    return outputs;
}

int ff_reconstruction(genome* g, int** array, node* head, int len, ff** lists,int* size, int* global_j){
    int i,j,k,flag = 1;
    
    /*controllo se è un input il nodo corrente*/
    if(head->innovation_number-1 < g->number_input){
        if((*size) == 0)
            (*lists) = (ff*)malloc(sizeof(ff));
        else
            (*lists) = (ff*)realloc((*lists),sizeof(ff)*(*size+1));
        
        (*lists)[(*size)].list_nodes = (node**)malloc(sizeof(node*)*len);
        (*lists)[(*size)].list_connections = (connection**)malloc(sizeof(connection*)*(len-1));
        (*lists)[(*size)].list_nodes[len-1] = head;
        (*lists)[(*size)].flag = 0;
        (*lists)[(*size)].size = len;
        (*size)++;
        
        return (*size);
    }
    
    /*controllo se l'ho già beccato*/
    if((*array)[head->innovation_number-1]){
        if((*size) == 0)
            (*lists) = (ff*)malloc(sizeof(ff));
        else
            (*lists) = (ff*)realloc((*lists),sizeof(ff)*(*size+1));
        
        (*lists)[(*size)].list_nodes = (node**)malloc(sizeof(node*)*len);
        (*lists)[(*size)].list_connections = (connection**)malloc(sizeof(connection*)*(len-1));
        (*lists)[(*size)].list_nodes[len-1] = head;
        (*lists)[(*size)].flag = 1;
        (*lists)[(*size)].size = len;
        (*size)++;
        return (*size);
    }
    
    /*controllo se è collegato a qualcosa attivamente*/
    for(i = 0; i < head->in_conn_size; i++){
        if(head->in_connections[i]->flag){
            flag = 0;
            break;
        }
    }
    
    if(flag){
        if((*size) == 0)
            (*lists) = (ff*)malloc(sizeof(ff));
        else
            (*lists) = (ff*)realloc((*lists),sizeof(ff)*(*size+1));
        
        (*lists)[(*size)].list_nodes = (node**)malloc(sizeof(node*)*len);
        (*lists)[(*size)].list_connections = (connection**)malloc(sizeof(connection*)*(len-1));
        (*lists)[(*size)].list_nodes[len-1] = head;
        (*lists)[(*size)].flag = -1;
        (*lists)[(*size)].size = len;
        (*size)++;
        return (*size);
    }
    j = (*global_j);
    /*non è nessuna delle tre quindi vado avanti, dico che l'ho trovato*/
    (*array)[head->innovation_number-1] = 1;
    for(i = 0; i < head->in_conn_size; i++){
        if(head->in_connections[i]->flag){
            k = ff_reconstruction(g, array, head->in_connections[i]->in_node,len+1,lists,size,global_j);
            if(k){
                for(;j < k; j++){
                    (*lists)[j].list_nodes[len-1] = head;
                    (*lists)[j].list_connections[len-1] = head->in_connections[i];
                }
                (*global_j) = k;
                j = k;
            }
        }
    }
    
    (*array)[head->innovation_number-1] = 0;

    
    return j;
    
}

int recursive_computation(int** array, node* head, genome* g, connection* c,float* actual_value){
    /*caso base stiamo su un estremo che può essere o un fake input o un input che prima era un 
     * fake input o un vero input*/
    int i,j;
    
    if(head->innovation_number-1 < g->number_input || c==NULL)
        head->flag = 1;
    else
        head->flag = 0;
        
    if(c!=NULL){
    //verifichiamo se stiamo in un input

        if(head->innovation_number-1 < g->number_input){
            c->flag = -2;
            (*actual_value)+=head->actual_value*c->weight;
            return 1;
        }
        
        else if((*array)[head->innovation_number-1]){
            return 0;
        }
        

    }
    
    for(i = 0; i < head->in_conn_size; i++){
        if(head->in_connections[i]->flag == -1){
            if(!head->flag)
                head->flag = recursive_computation(array,head->in_connections[i]->in_node,g,head->in_connections[i],&head->actual_value);
            else
                recursive_computation(array,head->in_connections[i]->in_node,g,head->in_connections[i],&head->actual_value);
        }
    }
    
    
    if(c!=NULL){
        if(head->flag){
            c->flag = -2;
            c->out_node->flag = 1;
            if(head->innovation_number-1 >= g->number_input)
                    (*actual_value)+=(modified_sigmoid(head->actual_value)*(c->weight));
            else
                (*actual_value)+=head->actual_value*head->out_connections[i]->weight;
            }
    }
 
    
    
    for(i = 0; i < head->out_conn_size; i++){
        if(head->out_connections[i]->flag == -1){
            head->out_connections[i]->flag = -2;
            if(head->flag){
                if(head->innovation_number-1 >= g->number_input)
                        head->out_connections[i]->out_node->actual_value+=(modified_sigmoid(head->actual_value)*(head->out_connections[i]->weight));
                    
                else
                    head->out_connections[i]->out_node->actual_value+=head->actual_value*head->out_connections[i]->weight;
                
                recursive_computation(array,head->out_connections[i]->out_node,g,NULL,&(head->out_connections[i]->out_node->actual_value));

                }
        }
    }

    return head->flag;
    
    
}


