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


// init parameters struct
params* init_params(int size, int input_size, int dimension1, int dimension2, int dimension3){
    params* p = (params*)malloc(sizeof(params));
    
    float* v = NULL;
    p->single_p = signed_kaiming_constant((float)(input_size));
    //p->single_p = random_general_gaussian_xavier_init((float)(input_size));
    if(size != 1){
        v = (float*)calloc(size,sizeof(float));
        int i;
        for(i = 0; i < size; i++){
            v[i] = random_general_gaussian(0,1);
        }
    }
    p->input_size = input_size;
    p->size = size;
    p->dimension1 = dimension1;
    p->dimension2 = dimension2;
    p->dimension3 = dimension3;
    p->p = v;
    return p;
}

void copy_param(params* original, params* copy){
    if(original == NULL || copy == NULL)
        return;
    if(original->size != copy->size)
        return;
    copy->single_p = original->single_p;
    if(original->size != 1){
        copy_array(original->p, copy->p, original->size);
    }
}

// init activation struct
activation* init_activation(int activation_flag){
    activation* a = (activation*)malloc(sizeof(activation));
    a->activation_flag = activation_flag;
    return a;
}


// init fcl_func struct
fcl_func* init_fcl_func(int input, int output){
    fcl_func* f = (fcl_func*)malloc(sizeof(fcl_func));
    f->input_size = input;
    f->output_size = output;
    f->v = (float*)calloc(output, sizeof(float));
    return f;
}

// init an edge
aco_edge* init_aco_edge(aco_node* input, aco_node* output, int operation_flag){
    aco_edge* e = (aco_edge*)malloc(sizeof(aco_edge));
    e->input = input;
    e->output = output;
    e->operation_flag = operation_flag;
    e->pheromone = 0;
    e->flag = 0;
    return e;
}


// init a node (only 1 struct must be passed)
aco_node* init_aco_node(params* weights, params* biases, activation* a, fcl_func* f){
    aco_node* n = (aco_node*)malloc(sizeof(aco_node));
    n->pheromone = 0;
    n->weights = weights;
    int i;
    
    if(weights != NULL){
        n->best_weights = init_params(n->weights->size,n->weights->input_size, n->weights->dimension1, n->weights->dimension2,n->weights->dimension3);
        n->velocity = init_params(n->weights->size,n->weights->input_size, n->weights->dimension1, n->weights->dimension2,n->weights->dimension3);
        n->best_weights->single_p = n->weights->single_p;
        n->velocity->single_p = 0;
        if(weights->size != 1){
            copy_array(n->weights->p,n->best_weights->p,n->weights->size);
            set_vector_with_value(0,n->velocity->p,n->velocity->size);
        }
    }
    
    else{
        n->velocity = NULL;
        n->best_weights = NULL;
    }
    
    n->biases = biases;
    
    if(biases != NULL){
        n->best_biases = init_params(biases->size,biases->input_size, biases->dimension1, biases->dimension2,biases->dimension3);
        n->velocity = init_params(biases->size,biases->input_size, biases->dimension1, biases->dimension2,biases->dimension3);
        n->best_biases->single_p = n->biases->single_p;
        n->velocity->single_p = 0;
        if(biases->size != 1){
            copy_array(n->biases->p,n->best_biases->p,n->biases->size);
            set_vector_with_value(0,n->velocity->p,n->velocity->size);
        }
    }
    
    else{
        n->best_biases = NULL;
    }
    n->a = a;
    n->f = f;
    n->flag = 0;
    n->input_pheromone = 0;
    n->output_pheromone = 0;
    n->best_personal = 0;
    n->best_global = 0;
    n->n_inputs = 0;
    n->inputs = NULL;
    n->n_outputs = 0;
    n->outputs = NULL;
    n->v = 0;
    n->best_global_params = NULL;
    return n;
}

// free params
void free_params(params* p){
    if(p == NULL)
        return;
    free(p->p);
    free(p);
}

// free activation just use free(a);

// free fcl_func
void free_fcl_func(fcl_func* f){
    if(f == NULL)
        return;
    free(f->v);
    free(f);
}

// free edge just free(e);

// not free of edges
void free_aco_node(aco_node* n){
    if(n == NULL)
        return;
    free(n->a);
    free_params(n->weights);
    free_params(n->biases);
    free_params(n->best_weights);
    free_params(n->best_biases);
    free_params(n->velocity);
    free_fcl_func(n->f);
    free(n->inputs);
    free(n->outputs);
    free(n);
}

// in v the output is saved and reset will reset the output
void reset_fcl_func(fcl_func* f){
    if(f == NULL)
        return;
    set_vector_with_value(0,f->v,f->output_size);
}

// reset edge: pheromone and flag
void reset_aco_edge(aco_edge* e){
    e->pheromone = 0;
    e->flag = 0;
}


void reset_aco_node(aco_node* n){
    reset_fcl_func(n->f);
    n->input_pheromone = 0;
    n->output_pheromone = 0;
    n->flag = 0;
}

// add an edge to a node
void add_aco_edge(aco_node* n, aco_edge* e, int input_edge){
    if(input_edge){
        e->output = n;
        if(n->inputs == NULL){
            n->inputs = (aco_edge**)malloc(sizeof(aco_edge*));
            n->inputs[0] = e;
        }
        else{
            n->inputs = (aco_edge**)realloc(n->inputs, sizeof(aco_edge*)*(n->n_inputs+1));
            n->inputs[n->n_inputs] = e;
        }
        n->n_inputs++;
        return;
    }
    e->input = n;
    if(n->outputs == NULL){
        n->outputs = (aco_edge**)malloc(sizeof(aco_edge*));
        n->outputs[0] = e;
    }
    else{
        n->outputs = (aco_edge**)realloc(n->outputs, sizeof(aco_edge*)*(n->n_outputs+1));
        n->outputs[n->n_outputs] = e;
    }
    n->n_outputs++;
    return;
}

// return the type of node
int node_state(aco_node* n){
    if(n->weights != NULL)
        return ACO_IS_WEIGHT;
    if(n->biases != NULL)
        return ACO_IS_BIAS;
    if(n->a != NULL)
        return ACO_IS_ACTIVATION;
    if(n->f != NULL)
        return ACO_IS_FCL;
}

aco_tracker* init_aco_tracker(){
    aco_tracker* a = (aco_tracker*)malloc(sizeof(aco_tracker));
    a->input_size = 0;
    a->weights_size = 0;
    a->biases_size = 0;
    a->n_fcl = 0;
    a->n_cl = 0;
    a->layers = 0;
    a->levy_threshold = 0.8;
    a->levy_ratio = 9.5;
    a->current_node = NULL;
    a->edge_taken = NULL;
    a->current_input = NULL;
    a->weights = NULL;
    a->biases = NULL;
    a->fcls = NULL;
    a->cls = NULL;
    a->m = NULL;
    return a;
}

void reset_aco_tracker(aco_tracker* a, int free_arrays, int free_model_flag){
    a->input_size = 0;
    a->weights_size = 0;
    a->biases_size = 0;
    a->n_fcl = 0;
    a->n_cl = 0;
    a->layers = 0;
    a->current_node = NULL;
    a->edge_taken = NULL;
    if(free_arrays){
        free(a->current_input);
        free(a->weights);
        free(a->biases);
    }
    if(free_model_flag){
        free_model(a->m);
    }
    a->current_input = NULL;
    a->weights = NULL;
    a->biases = NULL;
    a->fcls = NULL;
    a->cls = NULL;
    a->m = NULL;
}
