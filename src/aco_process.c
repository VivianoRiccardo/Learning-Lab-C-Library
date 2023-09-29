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

// first build the graph
// secondly build m models where m is the number of ants
// assign to each model a pheromone quantity
// clips the pheromone within min and max
// update pheromone trails with best model
// free models
// repeat

/*
 * 
 * Inputs:
 * 
 *             @ input_size:= is the size of the input for each model
 *             @ output_size:= is the size of the output of each model
 *             @ int** hidden_sizes:= is all the dimensions of each layer is exactly n_layers X hidden_sizes_length
 *                                   for the input as well as the output each element should be equal to input size or output size respectively
 *             @ int* hidden_widths:= is the width for each layer, dimension: n_layers
 *             @ int* hidden_depths:= is the depth for each layer, dimension: n_layers
 *             @ int_hidden_sizes_length:= is the number of possible fcl nodes for each layer
 *             @ int ants:= is the number of ants that want to use during the iteration (as well as the number of models that will be built)
 *             @ int number_of_iterations:= is th number of iterations used by the algorithm
 *             @ int time_update_best_trail:= is how often the best trail globally will be used to update the pheromone graph
 *             @ float p:= is in [0,1] is for the pheromone update
 *             @ float init_tau_max:= is a high number chosen for initializing the tau clipping value for the min-max algorithm
 * @ activations:= -1:= all activations, otherwise only that activation
 * */
aco_struct* init_aco(int* sizes, int* widths, int* depths, int* sub_dimensions,int* activations,  int n_layers, int ants, int number_of_iterations, int time_update_best_trail, float p, float init_tau_max, float p_dec, int max_iterations_fa){
    int i,j;
    // checking that there is at least 1 hidden layer
    if(n_layers < 2){
        fprintf(stderr,"Error: n layers must be >= 2, with n layers == 2 there is no hidden layer!\n");
        exit(1);
    }
    // checking that the depth of each layer has at least depth >= 3
    for(i = 0; i < n_layers; i++){
        if(depths[i] < 3){
            fprintf(stderr,"Error: every layer must have at least depth >= 3!\n");
            exit(1);
        }
    }
    // checking that the parameter for p is between 0 and 1
    if(p < 0 || p > 1){
        fprintf(stderr,"Error: the update value must in range [0,1]\n");
        exit(1);
    } 
    aco_struct* s = (aco_struct*)malloc(sizeof(aco_struct));
    s->iteration_index = 0;
    s->number_of_iterations = number_of_iterations;
    s->time_for_update_best_trail = time_update_best_trail;
    s->p = p;
    s->tau_max = init_tau_max;
    s->tau_min = 0;
    s->length_best_trail = 0;
    s->pheromone_best_trail = 0;
    s->best_trail = NULL;
    aco_node*** nodes = (aco_node***)malloc(sizeof(aco_node**));
    aco_edge*** edges = (aco_edge***)malloc(sizeof(aco_edge**));
    nodes[0] = NULL;
    edges[0] = NULL;
    s->number_of_edges = 0;
    s->number_of_nodes = 0;
    s->longest_trail = 0;
    s->layers = n_layers;
    s->nodes_layers = (aco_node***)malloc(sizeof(aco_node**)*(n_layers-1));
    s->widths = (int*)calloc(n_layers,sizeof(int));
    s->depths = (int*)calloc(n_layers,sizeof(int));
    copy_int_array(widths,s->widths,n_layers);
    copy_int_array(depths,s->depths,n_layers);
    aco_node* root = init_aco_node(NULL,NULL,NULL,NULL);
    s->sizes = (int*)calloc(n_layers,sizeof(int));
    copy_int_array(sizes,s->sizes,n_layers);
    for(i = 0; i < n_layers-1; i++){
        s->longest_trail+=depths[i];
        s->nodes_layers[i] = attach_layer(root,sizes[i],sizes[i+1],widths[i],depths[i],sub_dimensions[i], activations[i]); 
    }
    compute_all_nodes_edges(root, nodes, edges, &s->number_of_nodes, &s->number_of_edges);
    s->nodes = nodes[0];
    s->edges = edges[0];
    free(nodes);
    free(edges);
    aco_tracker* tracker = init_aco_tracker();
    s->root = root;
    s->tracker = tracker;
    s->number_of_ants = ants;
    s->second_dimension_for_trails = (int*)calloc(ants,sizeof(int));
    s->pheromones = (double*)calloc(ants,sizeof(double));
    s->m = NULL;
    s->best_trail = NULL;
    s->trails = (int**)malloc(sizeof(int*)*ants);
    for(i = 0; i < ants; i++){
        s->trails[i] = NULL;
    }
    for(i = 0; i < s->number_of_edges; i++){
        s->edges[i]->pheromone = s->tau_max;
    }
    for(i = 0; i < s->number_of_nodes; i++){
        s->nodes[i]->pheromone = s->tau_max;
    }
    s->average = s->number_of_edges/s->number_of_nodes;
    s->p_dec = p_dec;
    // pso
    s->inertia_max = 0.99;
    s->inertia_min = 0.4;
    s->c1 = 0;
    s->c2 = 2.04;
    s->inertia = 0.94;
    s->v_max = 1;
    //fa
    s->percentage_of_fireflies = 0.8;// also for ewoa
    s->lambda = 1;
    s->beta_min = 0.2;
    s->beta_zero = 1;
    s->step_size = 0.4/(1.0+exp(-0.015*((float)max_iterations_fa)/3.0));
    s->softmax_temperature = 2;// also for eowa
    // gsa
    s->g_zero = 100;
    s->alpha = 20;
    s->omega = 10;
    s->t_c = 4.0/((float)max_iterations_fa);
    s->rp_max = 1.5;
    s->rp_min = 0.5;
    s->g = s->g_zero/(1.0+exp(s->alpha*(0-s->t_c)/((float)max_iterations_fa)));
    s->alpha_velocity = 0.1;// also for pso
    s->h_velocity = 0.05;// also for pso
    // woa
    s->alpha_woa = 2;
    s->beta_woa = 1;
    s->weight_woa = 1;
    return s;
    
}

void set_x_params(aco_struct* s, float inertia_max, float inertia_min, float c1, float c2, float inertia, float v_max, float percentage_of_fireflies, float lambda_value, float beta_min,
                  float beta_zero, float softmax_temperature, float g_zero, float alpha, float omega, float rp_max, float rp_min, float alpha_velocity, float h_velocity){
    // pso
    s->inertia_max = inertia_max;
    s->inertia_min = inertia_min;
    s->c1 = c1;
    s->c2 = c2;
    s->inertia = inertia;
    s->v_max = v_max;
    //fa
    s->percentage_of_fireflies = percentage_of_fireflies;
    s->lambda = lambda_value;
    s->beta_min = beta_min;
    s->beta_zero = beta_zero;
    s->softmax_temperature = softmax_temperature;
    
    // gsa
    s->g_zero = g_zero;
    s->alpha =alpha;
    s->omega =omega;
    s->rp_max = rp_max;
    s->rp_min = rp_min;
    s->alpha_velocity = alpha_velocity;
    s->h_velocity = h_velocity;
}

void free_aco_struct(aco_struct* s){
    int i;
    for(i = 0; i < s->number_of_nodes;  i++){
        free_aco_node(s->nodes[i]);
    }
    free(s->nodes);
    for(i = 0; i < s->layers-1; i++){
        free(s->nodes_layers[i]);
    }
    free(s->nodes_layers);
    free_matrix((void**)s->edges,s->number_of_edges);
    free(s->tracker);
    free(s->pheromones);
    free(s->best_trail);
    free_matrix((void**)s->trails, s->number_of_ants);
    free(s->second_dimension_for_trails);
    free(s->widths);
    free(s->depths);
    free(s->sizes);
    free(s);
    
    return;
}

void update_best_trail(aco_struct* s){
    int i;
    for(i = 0; i < s->number_of_ants; i++){
        if(s->pheromones[i] > s->pheromone_best_trail){
            s->pheromone_best_trail = s->pheromones[i];
            free(s->best_trail);
            s->best_trail = (int*)malloc(sizeof(int)*s->second_dimension_for_trails[i]);
            s->length_best_trail = s->second_dimension_for_trails[i];
            copy_int_array(s->trails[i],s->best_trail,s->second_dimension_for_trails[i]);
        } 
    }
}

void set_pheromone_best_trail(aco_struct* s, double pheromone){
    s->pheromone_best_trail = pheromone;
}

void update_taus(aco_struct* s){
    if((1.0/(1-s->p))*s->pheromone_best_trail > s->tau_max || s->iteration_index == 1){
        s->tau_max = (1.0/(1-s->p))*s->pheromone_best_trail;
        s->tau_min = s->tau_max*(1-s->p_dec)/((s->average-1)*s->p_dec);
        if(s->tau_min > s->tau_max)
            s->tau_min = s->tau_max;
    }
}

void set_pheromone_to_index(aco_struct* s, int index, double pheromone){
    if(index > s->number_of_ants)
        return;
    s->pheromones[index] = pheromone;
}

void update_path_with_pheromone(aco_struct* s, int* path, int length_path, double pheromone){
    s->tracker->current_node = s->root;
    update_aco_tracker_next_according_to_path(s->tracker,pheromone,path,0, length_path, s->p, s->tau_max,s->tau_min);
}

void update_path_with_pheromone_according_to_nodes(aco_struct* s, int* path, int length_path, double pheromone){
    s->tracker->current_node = s->root;
    update_aco_tracker_next_according_to_path_according_to_nodes(s->tracker,pheromone,path,0, length_path, s->p, s->tau_max,s->tau_min);
}

void update_path_with_pheromone_according_to_subnodes(aco_struct* s, int* path, int length_path, double pheromone, int init_node, int final_node){
    s->tracker->current_node = s->root;
    update_aco_tracker_next_according_to_path_according_to_subnodes(s->tracker,pheromone,path,0, length_path, s->p, s->tau_max,s->tau_min, init_node, final_node);
}

void update_path_with_pheromone_no_min_max(aco_struct* s, int* path, int length_path, double pheromone){
    s->tracker->current_node = s->root;
    update_aco_tracker_next_according_to_path_no_min_max(s->tracker,pheromone,path,0, length_path, s->p, s->tau_max,s->tau_min);
}


void update_pheromones(aco_struct* s){
    update_graph_pheromone_from_trail(s);
    int i;
    for(i = 0; i < s->number_of_edges; i++){
        if(!s->edges[i]->flag){
            s->edges[i]->pheromone*=s->p;
            if(s->edges[i]->pheromone < s->tau_min)
                s->edges[i]->pheromone = s->tau_min;
            if(s->edges[i]->pheromone > s->tau_max)
                s->edges[i]->pheromone = s->tau_max;
        }
        else{
            s->edges[i]->flag = 0;
        }
    }
    s->iteration_index++;
}

void update_pheromones_according_to_nodes(aco_struct* s){
    update_graph_pheromone_from_trail_according_to_nodes(s);
    int i;
    for(i = 0; i < s->number_of_nodes; i++){
        if(!s->nodes[i]->flag){
            s->nodes[i]->pheromone*=s->p;
            if(s->nodes[i]->pheromone < s->tau_min)
                s->nodes[i]->pheromone = s->tau_min;
            if(s->nodes[i]->pheromone > s->tau_max)
                s->nodes[i]->pheromone = s->tau_max;
        }
        else{
            s->nodes[i]->flag = 0;
        }
    }
    s->iteration_index++;
}

void update_pheromones_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage){
    if (init_percentage > final_percentage){
        fprintf(stderr,"Error final percentage must be > than init percentage\n");
        exit(1);
    }
    int starting_node = s->number_of_nodes*init_percentage;
    int final_node = s->number_of_nodes*final_percentage;
    update_graph_pheromone_from_trail_according_to_subnodes(s, init_percentage, final_percentage);
    int i;
    for(i = starting_node; i < final_node; i++){
        if(!s->nodes[i]->flag){
            s->nodes[i]->pheromone*=s->p;
            if(s->nodes[i]->pheromone < s->tau_min)
                s->nodes[i]->pheromone = s->tau_min;
            if(s->nodes[i]->pheromone > s->tau_max)
                s->nodes[i]->pheromone = s->tau_max;
        }
        else{
            s->nodes[i]->flag = 0;
        }
    }
    s->iteration_index++;
}

void update_pheromones_no_min_max(aco_struct* s){
    update_graph_pheromone_from_trail_no_min_max(s);
    int i;
    for(i = 0; i < s->number_of_edges; i++){
        if(!s->edges[i]->flag){
            s->edges[i]->pheromone*=s->p;
        }
        else{
            s->edges[i]->flag = 0;
        }
    }
    s->iteration_index++;
}

void update_pheromones_best(aco_struct* s){
        
    if(!(s->iteration_index%s->time_for_update_best_trail)){
            update_path_with_pheromone(s, s->best_trail, s->length_best_trail, s->pheromone_best_trail);
        
        int i;
        for(i = 0; i < s->number_of_edges; i++){
            if(!s->edges[i]->flag){
                s->edges[i]->pheromone*=s->p;
                if(s->edges[i]->pheromone < s->tau_min)
                    s->edges[i]->pheromone = s->tau_min;
            }
            else{
                s->edges[i]->flag = 0;
            }
        }
    }
}

void update_pheromones_best_according_to_nodes(aco_struct* s){
        
    if(!(s->iteration_index%s->time_for_update_best_trail)){
        update_path_with_pheromone_according_to_nodes(s, s->best_trail, s->length_best_trail, s->pheromone_best_trail);
        
        int i;
        for(i = 0; i < s->number_of_nodes; i++){
            if(!s->nodes[i]->flag){
                s->nodes[i]->pheromone*=s->p;
                if(s->nodes[i]->pheromone < s->tau_min)
                    s->nodes[i]->pheromone = s->tau_min;
            }
            else{
                s->nodes[i]->flag = 0;
            }
        }
    }
}

void update_pheromones_best_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage){
    if(init_percentage > final_percentage || init_percentage < 0 || final_percentage > 1){
        fprintf(stderr,"Error in init or final percentage!\n");
        exit(1);
    }    
    if(!(s->iteration_index%s->time_for_update_best_trail)){
        int init_node = s->number_of_nodes*init_percentage;
        int final_node = s->number_of_nodes*final_percentage;
        update_path_with_pheromone_according_to_subnodes(s, s->best_trail, s->length_best_trail, s->pheromone_best_trail, init_node, final_node);
        
        int i;
        for(i = init_node; i < final_node; i++){
            if(!s->nodes[i]->flag){
                s->nodes[i]->pheromone*=s->p;
                if(s->nodes[i]->pheromone < s->tau_min)
                    s->nodes[i]->pheromone = s->tau_min;
            }
            else{
                s->nodes[i]->flag = 0;
            }
        }
    }
}

void update_graph_pheromone_from_trail(aco_struct* s){
    int index = 0;
    double pheromone = s->pheromones[0];
    int i;
    for(i = 1; i < s->number_of_ants; i++){
        if(s->pheromones[i] > pheromone){
            pheromone = s->pheromones[i];
            index = i;
        }
    }
    update_path_with_pheromone(s, s->trails[index], s->second_dimension_for_trails[index], s->pheromones[index]);
}

void update_graph_pheromone_from_trail_according_to_nodes(aco_struct* s){
    int index = 0;
    double pheromone = s->pheromones[0];
    int i;
    for(i = 1; i < s->number_of_ants; i++){
        if(s->pheromones[i] > pheromone){
            pheromone = s->pheromones[i];
            index = i;
        }
    }
    update_path_with_pheromone_according_to_nodes(s, s->trails[index], s->second_dimension_for_trails[index], s->pheromones[index]);
}

void update_graph_pheromone_from_trail_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage){
    int index = 0;
    double pheromone = s->pheromones[0];
    int i;
    for(i = 1; i < s->number_of_ants; i++){
        if(s->pheromones[i] > pheromone){
            pheromone = s->pheromones[i];
            index = i;
        }
    }
    int init_node = s->number_of_nodes*init_percentage;
    int final_node = s->number_of_nodes*final_percentage;
    update_path_with_pheromone_according_to_subnodes(s, s->trails[index], s->second_dimension_for_trails[index], s->pheromones[index], init_node, final_node);
}

void update_graph_pheromone_from_trail_no_min_max(aco_struct* s){
    int i;
    for(i = 0; i < s->number_of_ants; i++){
        update_path_with_pheromone_no_min_max(s, s->trails[i], s->second_dimension_for_trails[i], s->pheromones[i]);
    }
    
}

void reset_flags_from_aco_struct(aco_struct* s){
    int i;
    for(i = 0; i < s->number_of_nodes; i++){
        s->nodes[i]->flag = 0;
    }
    for(i = 0; i < s->number_of_edges; i++){
        s->edges[i]->flag = 0;
    }
}

model* get_model_according_to_path(aco_struct* s, int index_path){
    reset_aco_tracker(s->tracker, 1, 0);
    s->tracker->current_node = s->root;
    int i;
    for(i = 0; i < s->second_dimension_for_trails[index_path]; i++){
        aco_tracker_next_by_index(s->tracker, s->trails[index_path][i]);
        //aco_tracker_build_model(s->tracker);
        //aco_tracker_build_model_complete(s->tracker);
        aco_tracker_build_model_complete2(s->tracker);
    }
    return build_model_from_tracker(s->tracker);    
}

model* get_model_according_to_path_debug(aco_struct* s, int index_path){
    reset_aco_tracker(s->tracker, 1, 0);
    s->tracker->current_node = s->root;
    int i;
    for(i = 0; i < s->second_dimension_for_trails[index_path]; i++){
        printf("A %d\n",s->trails[index_path][i]);
        aco_tracker_next_by_index(s->tracker, s->trails[index_path][i]);
        //aco_tracker_build_model(s->tracker);
        //aco_tracker_build_model_complete(s->tracker);
        printf("B %d\n",s->trails[index_path][i]);
        aco_tracker_build_model_complete2(s->tracker);
        printf("C %d\n",s->trails[index_path][i]);
        
    }
    printf("D %d\n",s->trails[index_path][i]);
        
    return build_model_from_tracker(s->tracker);    
}

model* get_best_model(aco_struct* s){
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    int i = 0, size = 0;
    while(i != -1){
        i = aco_tracker_best_next(t);
        if(i == -1)
            break;
        size++;
        //aco_tracker_build_model(t);
        aco_tracker_build_model_complete(t);
    }
    model* m = build_model_from_tracker(t);    
    free(t);
    return m;
}

void build_path_from_ant_index(aco_struct* s, int index){
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    free(s->trails[index]);
    s->trails[index] = NULL;
    int i = 0, size = 0;
    while(i != -1){
        i = aco_tracker_next_weithout_setting_flags(t);
        if(i == -1)
            break;
        size++;
        s->trails[index] = (int*)realloc(s->trails[index],sizeof(int)*size);
        s->trails[index][size-1] = i;
    }
    s->second_dimension_for_trails[index] = size;
    free(t);
}

void build_path_from_ant_index_according_to_nodes(aco_struct* s, int index){
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    free(s->trails[index]);
    s->trails[index] = NULL;
    int i = 0, size = 0;
    while(i != -1){
        i = aco_tracker_next_weithout_setting_flags_according_to_nodes(t);
        if(i == -1)
            break;
        size++;
        s->trails[index] = (int*)realloc(s->trails[index],sizeof(int)*size);
        s->trails[index][size-1] = i;
    }
    s->second_dimension_for_trails[index] = size;
    free(t);
}

void build_path_from_ant_index_according_to_subnodes(aco_struct* s, int index, float init_percentage, float final_percentage){
    int init_node = s->number_of_nodes*init_percentage;
    int final_node = s->number_of_nodes*final_percentage;
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    free(s->trails[index]);
    s->trails[index] = NULL;
    int i = 0, size = 0;
    while(i != -1){
        if(size >= init_node && size < final_node)
            i = aco_tracker_next_weithout_setting_flags_according_to_nodes(t);
        else
            i = aco_tracker_next_weithout_setting_flags_according_to_best(t);
        if(i == -1)
            break;
        size++;
        s->trails[index] = (int*)realloc(s->trails[index],sizeof(int)*size);
        s->trails[index][size-1] = i;
    }
    s->second_dimension_for_trails[index] = size;
    free(t);
}

void build_best_path_from_ant_index_according_to_subnodes(aco_struct* s, int index, float init_percentage, float final_percentage){
    int init_node = s->number_of_nodes*init_percentage;
    int final_node = s->number_of_nodes*final_percentage;
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    free(s->trails[index]);
    s->trails[index] = NULL;
    int i = 0, size = 0;
    while(i != -1){

        i = aco_tracker_next_weithout_setting_flags_according_to_best(t);
        if(i == -1)
            break;
        size++;
        s->trails[index] = (int*)realloc(s->trails[index],sizeof(int)*size);
        s->trails[index][size-1] = i;
    }
    s->second_dimension_for_trails[index] = size;
    free(t);
}


double get_stagnation(aco_struct* s){
    double gamma = 0;
    int i;
    for(i = 0; i < s->number_of_edges; i++){
        double ret1 = s->tau_max- s->edges[i]->pheromone;
        double ret2 = s->edges[i]->pheromone - s->tau_min;
        if(ret1 < ret2)
            gamma+=ret1;
        else
            gamma+=ret2;
    }
    return gamma/((double)(s->number_of_nodes)*(double)(s->number_of_nodes));
}

double get_stagnation_according_to_nodes(aco_struct* s){
    double gamma = 0;
    int i;
    for(i = 0; i < s->number_of_nodes; i++){
        double ret1 = s->tau_max- s->nodes[i]->pheromone;
        double ret2 = s->nodes[i]->pheromone - s->tau_min;
        if(ret1 < ret2)
            gamma+=ret1;
        else
            gamma+=ret2;
    }
    return gamma/((double)(s->number_of_nodes)*(double)(s->number_of_nodes));
}

double get_stagnation_according_to_subnodes(aco_struct* s, int init_node, int final_node){
    double gamma = 0;
    int i;
    for(i = init_node; i < final_node; i++){
        double ret1 = s->tau_max- s->nodes[i]->pheromone;
        double ret2 = s->nodes[i]->pheromone - s->tau_min;
        if(ret1 < ret2)
            gamma+=ret1;
        else
            gamma+=ret2;
    }
    return ((float)(final_node-init_node))/((float)(s->number_of_nodes))*(gamma/((double)(final_node-init_node)*(double)(final_node-init_node)));
}

void recompute_pheromones(aco_struct* s, double delta){
    int i;
    for(i = 0; i < s->number_of_edges; i++){
        s->edges[i]->pheromone = s->edges[i]->pheromone + delta*(s->tau_max-s->edges[i]->pheromone);
        if(s->edges[i]->pheromone > s->tau_max)
            s->edges[i]->pheromone = s->tau_max;
        if(s->edges[i]->pheromone < s->tau_min)
            s->edges[i]->pheromone = s->tau_min;
    }
}

void recompute_pheromones_according_to_nodes(aco_struct* s, double delta){
    int i;
    for(i = 0; i < s->number_of_nodes; i++){
        s->nodes[i]->pheromone = s->nodes[i]->pheromone + delta*(s->tau_max-s->nodes[i]->pheromone);
    }
}

void recompute_pheromones_according_to_subnodes(aco_struct* s, double delta, int init_node, int final_node){
    int i;
    for(i = init_node; i < final_node; i++){
        s->nodes[i]->pheromone = s->nodes[i]->pheromone + delta*(s->tau_max-s->nodes[i]->pheromone);
    }
}

double recompute_pheromones_according_to_stagnation(aco_struct* s, double stagnation_threshold, double delta){
    double stagnation = get_stagnation(s);
    if(stagnation < stagnation_threshold)
        recompute_pheromones(s,delta);
    return stagnation;
}

double recompute_pheromones_according_to_stagnation_according_to_nodes(aco_struct* s, double stagnation_threshold, double delta){
    double stagnation = get_stagnation_according_to_nodes(s);
    if(stagnation < stagnation_threshold)
        recompute_pheromones_according_to_nodes(s,delta);
    return stagnation;
}

double recompute_pheromones_according_to_stagnation_according_to_subnodes(aco_struct* s, double stagnation_threshold, double delta, float init_percentage, float final_percentage){
    int init_node = s->number_of_nodes*init_percentage;
    int final_node = s->number_of_nodes*final_percentage;
    double stagnation = get_stagnation_according_to_subnodes(s, init_node, final_node);
    if(stagnation < stagnation_threshold)
        recompute_pheromones_according_to_subnodes(s,delta, init_node, final_node);
    return stagnation;
}

void calculate_pso_function_per_node(aco_struct* s){
    int i,j;
    for(i = 0; i < s->number_of_nodes; i++){
        if(node_state(s->nodes[i]) == ACO_IS_WEIGHT || node_state(s->nodes[i]) == ACO_IS_BIAS){
            double fitness = 0;
            for(j = 0; j < s->nodes[i]->n_inputs; j++){
                fitness+=s->nodes[i]->inputs[j]->pheromone;
            }
            s->nodes[i]->input_pheromone = fitness;
        }
    }
}

void calculate_pso_function_per_node_according_to_nodes(aco_struct* s){
    int i,j;
    for(i = 0; i < s->number_of_nodes; i++){
        if(node_state(s->nodes[i]) == ACO_IS_WEIGHT || node_state(s->nodes[i]) == ACO_IS_BIAS){
            s->nodes[i]->input_pheromone = s->nodes[i]->pheromone;
        }
    }
}

void set_best_personal_node_fitness_pso(aco_struct* s){
    int i,j;
    for(i = 0; i < s->number_of_nodes; i++){
        if(node_state(s->nodes[i]) == ACO_IS_WEIGHT || node_state(s->nodes[i]) == ACO_IS_BIAS){
            if(s->nodes[i]->input_pheromone > s->nodes[i]->best_personal){
                s->nodes[i]->best_personal = s->nodes[i]->input_pheromone;
                copy_param(s->nodes[i]->weights,s->nodes[i]->best_weights);
                copy_param(s->nodes[i]->biases,s->nodes[i]->best_biases);
            } 
        }
    }
}

void set_best_global_node_fitness_pso(aco_struct* s){
    int i,j,k;
    for(i = 0; i < s->layers-1; i++){
        for(j = 0; j < 4*s->depths[i]-3; j++){
            double best_global = -1;
            params* best_params = NULL;
            for(k = 0; k < s->widths[i]; k++){
                if(s->nodes_layers[i][j*s->widths[i]+k]->best_personal >  best_global){
                    best_global = s->nodes_layers[i][j*s->widths[i]+k]->best_personal;
                    if(node_state(s->nodes_layers[i][j*s->widths[i]+k]) == ACO_IS_WEIGHT)
                        best_params = s->nodes_layers[i][j*s->widths[i]+k]->best_weights;
                    else if(node_state(s->nodes_layers[i][j*s->widths[i]+k]) == ACO_IS_BIAS)
                        best_params = s->nodes_layers[i][j*s->widths[i]+k]->best_biases;
                }
            }
            for(k = 0; k < s->widths[i]; k++){
                s->nodes_layers[i][j*s->widths[i]+k]->best_global = best_global;
                s->nodes_layers[i][j*s->widths[i]+k]->best_global_params = best_params;
            }
        }
    }
}

void set_best_global_node_fitness_pso_according_to_nodes(aco_struct* s){
    int i,j,k;
    for(i = 0; i < s->layers-1; i++){
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            double best_global = -1;
            params* best_params = NULL;
            for(k = 0; k < s->widths[i]; k++){
                if(s->nodes_layers[i][j*s->widths[i]+k]->best_personal >  best_global){
                    best_global = s->nodes_layers[i][j*s->widths[i]+k]->best_personal;
                    if(node_state(s->nodes_layers[i][j*s->widths[i]+k]) == ACO_IS_WEIGHT)
                        best_params = s->nodes_layers[i][j*s->widths[i]+k]->best_weights;
                    else if(node_state(s->nodes_layers[i][j*s->widths[i]+k]) == ACO_IS_BIAS)
                        best_params = s->nodes_layers[i][j*s->widths[i]+k]->best_biases;
                }
            }
            for(k = 0; k < s->widths[i]; k++){
                s->nodes_layers[i][j*s->widths[i]+k]->best_global = best_global;
                s->nodes_layers[i][j*s->widths[i]+k]->best_global_params = best_params;
            }
        }
    }
}

int aco_weights_are_different(float* p1, float* p2, int size){
    int i;
    for(i = 0; i < size; i++){
        if(p1[i] != p2[i])
            return 1;
    }
    return 0;
}

int get_width_according_to_node_index(aco_struct* s, int index){
    int i;
    int sum = 0;
    for(i = 0; i < s->layers-1; i++){
        sum+=s->sizes[i]*s->sizes[i+1];
        if (index < sum)
            return s->widths[i];
    }
}

void update_pso_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations){
    int i,j,k,w;
    for(i = 0; i < s->layers-1; i++){
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            
            /*float x_max = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
            float x_min = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
 
            
            for(k = 1; k < s->widths[i]; k++){
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p > x_max)
                    x_max = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p < x_min)
                    x_min = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
            }
            
            double v_max = s->alpha_velocity*(x_max-x_min);
            v_max *= (1.0-pow(current_iteration/max_number_of_iterations,s->h_velocity));
            */
            for(k = 0; k < s->widths[i]; k++){
                if(s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p != s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p){
                    s->nodes_layers[i][j*s->widths[i]+k]->v = s->inertia*s->nodes_layers[i][j*s->widths[i]+k]->v + s->c1*r2()*(s->nodes_layers[i][j*s->widths[i]+k]->best_weights->single_p - s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p) + s->c2*r2()*(s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p - s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                    if(s->nodes_layers[i][j*s->widths[i]+k]->v > s->v_max)
                        s->nodes_layers[i][j*s->widths[i]+k]->v = s->v_max;
                    if(s->nodes_layers[i][j*s->widths[i]+k]->v < -s->v_max)
                        s->nodes_layers[i][j*s->widths[i]+k]->v = -s->v_max;
                    s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p += s->nodes_layers[i][j*s->widths[i]+k]->v;
                }
            }
        }
    }
    reset_flags_from_aco_struct(s);
}

void update_woa_nodes(aco_struct* s){
    int i,j,k,w;
    for(i = 0; i < s->layers-1; i++){
        
        float* values = (float*)calloc(s->widths[i],sizeof(float));
        int* indices = (int*)calloc(s->widths[i],sizeof(int));
        
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            
            
            double sum = 0;
            
            for(k = 0; k < s->widths[i]; k++){
                values[k] = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                indices[k] = k;
            }
            sort(values,indices,0,s->widths[i]-1);

            for(k = (int)(s->widths[i]*s->percentage_of_fireflies); k < s->widths[i]; k++){
                values[indices[k]] = pow(values[indices[k]],1.0/s->softmax_temperature);
                sum+=values[indices[k]];
            }
            for(k = (int)(s->widths[i]*s->percentage_of_fireflies); k < s->widths[i]; k++){
                values[indices[k]]/=sum;
            }
            
            for(k = 0; k < s->widths[i]; k++){
                if(s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p != s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p){
                    float a = s->alpha_woa*2*r2()-s->alpha_woa;
                    float c = 2*r2();
                    float d = float_abs(s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                    if (r2() < 0.5){
                        if(float_abs(a) < 1){
                            d = float_abs(c*s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                            s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p = s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p-s->weight_woa*a*d;
                        }
                        else{
                            float val = s->widths[i]*r2();
                            int index = ((int)(val));
                            if (index == s->widths[i])
                                index--;
                            
                            float ran = r2();
                            for(w = (int)(s->widths[i]*s->percentage_of_fireflies); w < s->widths[i]; w++){
                                if(ran <= values[indices[w]]){
                                    index = indices[w];
                                    break;
                                }
                                else
                                    ran-=values[indices[w]];
                            }
                            
                            
                            d = float_abs(c*s->nodes_layers[i][j*s->widths[i]+index]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                            s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p = s->nodes_layers[i][j*s->widths[i]+index]->weights->single_p-s->weight_woa*a*d;
                        }
                    }
                    else{
                        float value = r2();
                        if (r2() < 0.5)
                            value*=-1;
                        s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p = s->weight_woa*d*exp(s->beta_woa*value)*cos(2*PI*value) + s->nodes_layers[i][j*s->widths[i]+k]->best_global_params->single_p;
                    }
                }
            }
        }
        free(values);
        free(indices);
    }
    reset_flags_from_aco_struct(s);
}

void update_fa_nodes(aco_struct* s){
    
    int i,j,k,w;
    for(i = 0; i < s->layers-1; i++){
        float* values = (float*)calloc(s->widths[i],sizeof(float));
        int* indices = (int*)calloc(s->widths[i],sizeof(int));
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            double sum = 0;
            
            for(k = 0; k < s->widths[i]; k++){
                values[k] = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                indices[k] = k;
            }
            sort(values,indices,0,s->widths[i]-1);

            for(k = (int)(s->widths[i]*s->percentage_of_fireflies); k < s->widths[i]; k++){
                values[indices[k]] = pow(values[indices[k]],1.0/s->softmax_temperature);
                sum+=values[indices[k]];
            }
            for(k = (int)(s->widths[i]*s->percentage_of_fireflies); k < s->widths[i]; k++){
                values[indices[k]]/=sum;
            }
            
            for(k = 0; k < (int)(s->widths[i]*s->percentage_of_fireflies); k++){
                float ran = r2();
                for(w = (int)(s->widths[i]*s->percentage_of_fireflies); w < s->widths[i]; w++){
                    if(ran <= values[indices[w]]){
                        if(s->nodes_layers[i][j*s->widths[i]+indices[k]]->input_pheromone < s->nodes_layers[i][j*s->widths[i]+indices[w]]->input_pheromone){
                            double difference = s->nodes_layers[i][j*s->widths[i]+indices[k]]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+indices[w]]->weights->single_p;
                            difference*=difference;
                            s->nodes_layers[i][j*s->widths[i]+indices[k]]->weights->single_p+=s->step_size*(r2()-1.0/2.0)+(s->nodes_layers[i][j*s->widths[i]+indices[w]]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+indices[k]]->weights->single_p)*((s->beta_min) + (s->beta_zero-s->beta_min)*exp(-s->lambda*difference));
                        }
                        break;
                    }
                    else
                        ran-=values[indices[w]];
                }
            }
        }
        free(values);
        free(indices);
    }
    reset_flags_from_aco_struct(s);
}

void update_gsa_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations){
    
    int i,j,k,w;
    for(i = 0; i < s->layers-1; i++){
        float* masses = (float*)calloc(s->widths[i],sizeof(float));
        float* deltas = (float*)calloc(s->widths[i],sizeof(float));
        float* forces = (float*)calloc(s->widths[i],sizeof(float));
        
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            //printf("new\n");
            double f_min = s->nodes_layers[i][j*s->widths[i]]->input_pheromone;
            double x_min = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
            double x_max = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
            double f_max = s->nodes_layers[i][j*s->widths[i]]->input_pheromone;
            for(k = 0; k < s->widths[i]; k++){
                for(w = 0;w < s->widths[i]; w++){
                    float distance = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p;
                    if(distance < 0)
                        distance*=-1;
                    deltas[k] += distance;
                }
                deltas[k]/=((float)s->widths[i]-1);
                if(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone < f_min)
                    f_min = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                if(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone > f_max)
                    f_max = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p > x_max)
                    x_max = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p < x_min)
                    x_min = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
            }
            //printf("min: %f, max: %f\n",f_min,f_max);
            double sum = 0;
            for(k = 0; k < s->widths[i]; k++){
                //printf("pheromone: %f, result minmax: %f\n",s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone,(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone-f_min)/(f_max-f_min));
                masses[k] = (s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone-f_min)/(f_max-f_min);
                sum+=masses[k];
            }
            
            for(k = 0; k < s->widths[i]; k++){
                masses[k]/=sum;
                //printf("mass: %f, deltas: %f\n",masses[k],deltas[k]);
            }
            
            for(k = 0; k < s->widths[i]; k++){
                double rp;
                if(deltas[k] < 1){
                    rp = s->rp_min+(s->rp_max-s->rp_min)*exp(1.0-1.0/deltas[k]);
                }
                else{
                    rp = s->rp_min+(s->rp_max-s->rp_min)*exp(1.0/deltas[k]);
                }
                for(w = 0;w < s->widths[i]; w++){
                    float distance = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p;
                    if(distance < 0)
                        distance*=-1;
                    forces[k] += r2()*(s->g*masses[w]/(pow(distance,rp)+EPSILON))*(s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                    //printf("mass k: %f, forces: %f, rp: %f\n",masses[k],forces[k],rp);
                }
            }
            
            
            double v_max = s->alpha_velocity*(x_max-x_min);
            v_max *= (1.0-pow(current_iteration/max_number_of_iterations,s->h_velocity));
            for(k = 0; k < s->widths[i]; k++){
                //double intelligent = r2()*(x_max-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p)/float_abs(0.7*x_min - s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                double intelligent = 0;
                double val = forces[k];
                s->nodes_layers[i][j*s->widths[i]+k]->v = r2()*s->nodes_layers[i][j*s->widths[i]+k]->v + val;
                if(s->nodes_layers[i][j*s->widths[i]+k]->v > v_max)
                    s->nodes_layers[i][j*s->widths[i]+k]->v = v_max;
                if(s->nodes_layers[i][j*s->widths[i]+k]->v < -v_max)
                    s->nodes_layers[i][j*s->widths[i]+k]->v = -v_max;
                //printf("forces: %f, masses: %f, fitnesses:%f, acceleration: %f, velocity: %f, ex pos: %f, exponent: %f, randomness: %f, new pos: %f\n",forces[k],masses[k],s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone,val, s->nodes_layers[i][j*s->widths[i]+k]->v, s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p, exp(pow(-current_iteration/max_number_of_iterations,s->omega)),(1.0 + (current_iteration/(max_number_of_iterations + random_beta(2, 5)))),exp(pow(-current_iteration/max_number_of_iterations,s->omega))*s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p + (1.0 + (current_iteration/(max_number_of_iterations + random_beta(2, 5))))*s->nodes_layers[i][j*s->widths[i]+k]->v);
                s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p = intelligent + exp(pow(-current_iteration/max_number_of_iterations,s->omega))*s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p + (1.0 + (current_iteration/(max_number_of_iterations + random_beta(2, 5))))*s->nodes_layers[i][j*s->widths[i]+k]->v;
            }
            //exit(0);
            set_vector_with_value(0,masses,s->widths[i]);
            set_vector_with_value(0,forces,s->widths[i]);
            set_vector_with_value(0,deltas,s->widths[i]);
        }
        free(masses);
        free(forces);
        free(deltas);
    }
    //exit(0);
    reset_flags_from_aco_struct(s);
}

void set_iteration_index_to_value(aco_struct* s, int value){
    s->iteration_index = value;
}

void update_psogsa_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations){
    
    int i,j,k,w;
    for(i = 0; i < s->layers-1; i++){
        float* masses = (float*)calloc(s->widths[i],sizeof(float));
        float* deltas = (float*)calloc(s->widths[i],sizeof(float));
        float* forces = (float*)calloc(s->widths[i],sizeof(float));
        
        for(j = 0; j < s->sizes[i]*s->sizes[i+1]; j++){
            //printf("new\n");
            double f_min = s->nodes_layers[i][j*s->widths[i]]->input_pheromone;
            double x_min = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
            double x_max = s->nodes_layers[i][j*s->widths[i]]->weights->single_p;
            double f_max = s->nodes_layers[i][j*s->widths[i]]->input_pheromone;
            for(k = 0; k < s->widths[i]; k++){
                for(w = 0;w < s->widths[i]; w++){
                    float distance = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p;
                    if(distance < 0)
                        distance*=-1;
                    deltas[k] += distance;
                }
                deltas[k]/=((float)s->widths[i]-1);
                if(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone < f_min)
                    f_min = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                if(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone > f_max)
                    f_max = s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone;
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p > x_max)
                    x_max = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p < x_min)
                    x_min = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p;
            }
            //printf("min: %f, max: %f\n",f_min,f_max);
            double sum = 0;
            for(k = 0; k < s->widths[i]; k++){
                //printf("pheromone: %f, result minmax: %f\n",s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone,(s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone-f_min)/(f_max-f_min));
                masses[k] = (s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone-f_min)/(f_max-f_min);
                sum+=masses[k];
            }
            
            for(k = 0; k < s->widths[i]; k++){
                masses[k]/=sum;
                //printf("mass: %f, deltas: %f\n",masses[k],deltas[k]);
            }
            
            for(k = 0; k < s->widths[i]; k++){
                double rp;
                if(deltas[k] < 1){
                    rp = s->rp_min+(s->rp_max-s->rp_min)*exp(1.0-1.0/deltas[k]);
                }
                else{
                    rp = s->rp_min+(s->rp_max-s->rp_min)*exp(1.0/deltas[k]);
                }
                for(w = 0;w < s->widths[i]; w++){
                    float distance = s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p;
                    if(distance < 0)
                        distance*=-1;
                    forces[k] += r2()*(s->g*masses[w]/(pow(distance,rp)+EPSILON))*(s->nodes_layers[i][j*s->widths[i]+w]->weights->single_p-s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                    //printf("mass k: %f, forces: %f, rp: %f\n",masses[k],forces[k],rp);
                }
            }
            
            double v_max = s->alpha_velocity*(x_max-x_min);
            v_max *= (1.0-pow(current_iteration/max_number_of_iterations,s->h_velocity));
            for(k = 0; k < s->widths[i]; k++){
                double val = forces[k];
                if(s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p != x_max){
                    s->nodes_layers[i][j*s->widths[i]+k]->v = s->inertia*s->nodes_layers[i][j*s->widths[i]+k]->v + r2()*s->c1*val + r2()*s->c2*(x_max - s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p);
                    if(s->nodes_layers[i][j*s->widths[i]+k]->v > v_max)
                        s->nodes_layers[i][j*s->widths[i]+k]->v = v_max;
                    if(s->nodes_layers[i][j*s->widths[i]+k]->v < -v_max)
                        s->nodes_layers[i][j*s->widths[i]+k]->v = -v_max;
                    //printf("forces: %f, masses: %f, fitnesses:%f, acceleration: %f, velocity: %f, ex pos: %f, exponent: %f, randomness: %f, new pos: %f\n",forces[k],masses[k],s->nodes_layers[i][j*s->widths[i]+k]->input_pheromone,val, s->nodes_layers[i][j*s->widths[i]+k]->v, s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p, exp(pow(-current_iteration/max_number_of_iterations,s->omega)),(1.0 + (current_iteration/(max_number_of_iterations + random_beta(2, 5)))),exp(pow(-current_iteration/max_number_of_iterations,s->omega))*s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p + (1.0 + (current_iteration/(max_number_of_iterations + random_beta(2, 5))))*s->nodes_layers[i][j*s->widths[i]+k]->v);
                    s->nodes_layers[i][j*s->widths[i]+k]->weights->single_p += s->nodes_layers[i][j*s->widths[i]+k]->v;
                }
            }
            //exit(0);
            set_vector_with_value(0,masses,s->widths[i]);
            set_vector_with_value(0,forces,s->widths[i]);
            set_vector_with_value(0,deltas,s->widths[i]);
        }
        free(masses);
        free(forces);
        free(deltas);
    }
    //exit(0);
    reset_flags_from_aco_struct(s);
}

float get_lambda_pso(int max_number_of_iterations, int current_iteration){
    if(current_iteration < 0.75*max_number_of_iterations)
        return r2();
}

float get_lvalue_pso(int max_number_of_iterations, int current_iteration){
    if(current_iteration < 0.75*max_number_of_iterations)
        return 4*r2();
    return -4;
}

void update_gsa_params(aco_struct* s, float current_iteration, float max_iterations){
    s->g = s->g_zero/(1.0+exp(s->alpha*(current_iteration-s->t_c)/((float)max_iterations)));
}
    
void update_fa_params(aco_struct* s, float current_iteration, float max_iterations){
    s->step_size = 0.4/(1.0+exp(0.015*(current_iteration- max_iterations)/3.0));
}
void update_pso_params(aco_struct* s, float current_iteration, float max_iterations){
    s->inertia-=s->inertia_max - (s->inertia_max-s->inertia_min)*2.0*(current_iteration/max_iterations);
    if (s->inertia < s->inertia_min)
        s->inertia = s->inertia_min;
    if (s->inertia > s->inertia_max)
        s->inertia = s->inertia_max;
    s->c2 = -0.7*cos(3.14*current_iteration/max_iterations)+2.04;
    s->c1 = -1.5*cos(3.14*current_iteration/max_iterations)+2.8;
    
}
void update_woa_params(aco_struct* s, float current_iteration, float max_iterations){
    s->alpha_woa = 2.0 - 2.0*current_iteration/max_iterations;
    double temp = current_iteration/max_iterations;
    temp*=temp*temp;
    s->weight_woa = 1.0 - 2.0*temp;
}
void update_psogsa_params(aco_struct* s, float current_iteration, float max_iterations){
    s->inertia-=s->inertia_max - (s->inertia_max-s->inertia_min)*2.0*(current_iteration/max_iterations);
    if (s->inertia < s->inertia_min)
        s->inertia = s->inertia_min;
    if (s->inertia > s->inertia_max)
        s->inertia = s->inertia_max;
    s->c2 = -0.7*cos(3.14*current_iteration/max_iterations)+2.04;
    s->c1 = -1.5*cos(3.14*current_iteration/max_iterations)+2.8;
    s->g = s->g_zero/(1.0+exp(s->alpha*(current_iteration-s->t_c)/((float)max_iterations)));
}


