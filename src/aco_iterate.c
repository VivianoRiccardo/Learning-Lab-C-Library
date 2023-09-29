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


void aco_tracker_execute_operation(aco_tracker* t){
    if(node_state(t->current_node) == ACO_IS_WEIGHT){// operations about weights [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying weights
            free(t->weights);
            t->weights = (float*)calloc(t->current_node->weights->size,sizeof(float));
            copy_array(t->current_node->weights->p,t->weights,t->current_node->weights->size);
            t->weights_size = t->current_node->weights->size;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MUL){// multiply weights
            if(t->current_node->weights->size == 1){// by single value
                mul_value(t->weights, t->current_node->weights->single_p, t->weights, t->weights_size);
            } 
            else{// by same matrix dimension
                dot1D(t->weights,t->current_node->weights->p,t->weights, t->weights_size);
            }
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_SUM){// sum weights
            if(t->current_node->weights->size == 1){// by single value
                add_value(t->weights, t->current_node->weights->single_p, t->weights, t->weights_size);
            } 
            else{// by same matrix dimension
                sum1D(t->weights,t->current_node->weights->p,t->weights, t->weights_size);
            }
        }
        
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MATRIX_MUL){// sum weights
                int size1 = t->weights_size/t->current_node->weights->dimension1;
                int size2 = t->current_node->weights->dimension1;
                int size3 = t->current_node->weights->dimension2;
                float* new_weights = (float*)calloc(size1*size3,sizeof(float));
                mat_mul(t->weights,t->current_node->weights->p, new_weights, size1,size2,size3);
                free(t->weights);
                t->weights = new_weights;
                t->weights_size = size1*size3;
            }
    }
    
    else if(node_state(t->current_node) == ACO_IS_BIAS){// operations about biases [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying biases
            free(t->biases);
            t->biases = (float*)calloc(t->current_node->biases->size,sizeof(float));
            copy_array(t->current_node->biases->p,t->biases,t->current_node->biases->size);
            t->biases_size = t->current_node->biases->size;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MUL){// multiply biases
            if(t->current_node->biases->size == 1){// by single value
                mul_value(t->biases, t->current_node->biases->single_p, t->biases, t->biases_size);
            } 
            else{// by same matrix dimension
                dot1D(t->biases,t->current_node->biases->p,t->biases, t->biases_size);
            }
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_SUM){// sum biases
            if(t->current_node->biases->size == 1){// by single value
                add_value(t->biases, t->current_node->biases->single_p, t->biases, t->biases_size);
            } 
            else{// by same matrix dimension
                sum1D(t->biases,t->current_node->biases->p,t->biases, t->biases_size);
            }
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_ACTIVATION){// just execution
        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            if(t->current_node->a->activation_flag == SIGMOID){
                sigmoid_array(t->current_input, t->current_input, t->input_size);
            }
            else if(t->current_node->a->activation_flag == RELU){
                relu_array(t->current_input, t->current_input, t->input_size);
            }
            else if(t->current_node->a->activation_flag == SOFTMAX){
                float* out = (float*)calloc(t->input_size,sizeof(float));
                softmax(t->current_input, out, t->input_size);
                free(t->current_input);
                t->current_input = out;
            }
            else if(t->current_node->a->activation_flag == TANH){
                tanhh_array(t->current_input, t->current_input, t->input_size);
            }
            else if(t->current_node->a->activation_flag == LEAKY_RELU){
                leaky_relu_array(t->current_input, t->current_input, t->input_size);
            }
            else if(t->current_node->a->activation_flag == ELU){
                elu_array(t->current_input, t->current_input, t->input_size, ELU_THRESHOLD);
            }
        }
    }
    else if(node_state(t->current_node) == ACO_IS_FCL){// just execution
        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            float* out = (float*)calloc(t->current_node->f->output_size,sizeof(float));
            fully_connected_feed_forward(t->current_input,out,t->weights,t->biases,t->current_node->f->input_size,t->current_node->f->output_size);
            t->input_size = t->current_node->f->output_size;
            free(t->current_input);
            t->current_input = out;
            free(t->weights);
            t->weights = NULL;
            free(t->biases);
            t->biases = NULL;
            t->weights_size = 0;
            t->biases_size = 0;
        }
    }
}

void aco_tracker_build_model_complete(aco_tracker* t){
    if(node_state(t->current_node) == ACO_IS_WEIGHT){// operations about weights [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying weights
            free(t->weights);
            free(t->biases);
            t->weights = (float*)calloc(t->current_node->weights->dimension1*t->current_node->weights->dimension2,sizeof(float));
            t->biases = (float*)calloc(t->current_node->weights->dimension2,sizeof(float));
            t->weights[t->current_node->weights->dimension3] = t->current_node->weights->single_p;
            t->weights_size = t->current_node->weights->dimension1*t->current_node->weights->dimension2;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_TAKE){// multiply weights
            t->weights[t->current_node->weights->dimension3] = t->current_node->weights->single_p;
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_ACTIVATION){// just execution
        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            t->fcls[t->n_fcl-1]->activation_flag = t->current_node->a->activation_flag;
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_FCL){// just execution
                

        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            if(t->fcls == NULL){
                t->fcls = (fcl**)malloc(sizeof(fcl*));
                t->fcls[0] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[0]);
                free(t->fcls[0]->weights);
                free(t->fcls[0]->biases);
                t->fcls[0]->weights = t->weights;
                t->fcls[0]->biases = t->biases;
            }
            else{
                t->fcls = (fcl**)realloc(t->fcls, sizeof(fcl*)*(t->n_fcl+1));
                t->fcls[t->n_fcl] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[t->n_fcl]);
                free(t->fcls[t->n_fcl]->weights);
                free(t->fcls[t->n_fcl]->biases);
                t->fcls[t->n_fcl]->weights = t->weights;
                t->fcls[t->n_fcl]->biases = t->biases;
            }
            t->weights = NULL;
            t->biases = NULL;
            t->weights_size = 0;
            t->biases_size = 0;
            t->n_fcl++;
            t->layers++;
        }
    }
}
void aco_tracker_build_model_complete2(aco_tracker* t){
    if(node_state(t->current_node) == ACO_IS_WEIGHT){// operations about weights [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying weights
            free(t->weights);
            free(t->biases);
            t->weights = (float*)calloc(t->current_node->weights->dimension1*t->current_node->weights->dimension2,sizeof(float));
            t->biases = (float*)calloc(t->current_node->weights->dimension2,sizeof(float));
            t->weights[t->current_node->weights->dimension3] = t->current_node->weights->single_p;
            t->weights_size = t->current_node->weights->dimension1*t->current_node->weights->dimension2;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_INDEX_COPY){// multiply weights
            t->weights[t->current_node->weights->dimension3] = t->current_node->weights->single_p;
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_ACTIVATION){// just execution
        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            t->fcls[t->n_fcl-1]->activation_flag = t->current_node->a->activation_flag;
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_FCL){// just execution
                

        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            if(t->fcls == NULL){
                t->fcls = (fcl**)malloc(sizeof(fcl*));
                t->fcls[0] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[0]);
                free(t->fcls[0]->weights);
                free(t->fcls[0]->biases);
                t->fcls[0]->weights = t->weights;
                t->fcls[0]->biases = t->biases;
            }
            else{
                t->fcls = (fcl**)realloc(t->fcls, sizeof(fcl*)*(t->n_fcl+1));
                t->fcls[t->n_fcl] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[t->n_fcl]);
                free(t->fcls[t->n_fcl]->weights);
                free(t->fcls[t->n_fcl]->biases);
                t->fcls[t->n_fcl]->weights = t->weights;
                t->fcls[t->n_fcl]->biases = t->biases;
            }
            t->weights = NULL;
            t->biases = NULL;
            t->weights_size = 0;
            t->biases_size = 0;
            t->n_fcl++;
            t->layers++;
        }
    }
}

void aco_tracker_build_model(aco_tracker* t){
    if(node_state(t->current_node) == ACO_IS_WEIGHT){// operations about weights [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying weights
            free(t->weights);
            t->weights = (float*)calloc(t->current_node->weights->size,sizeof(float));
            copy_array(t->current_node->weights->p,t->weights,t->current_node->weights->size);
            t->weights_size = t->current_node->weights->size;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MUL){// multiply weights
            if(t->current_node->weights->size == 1){// by single value
                mul_value(t->weights, t->current_node->weights->single_p, t->weights, t->weights_size);
            } 
            else{// by same matrix dimension
                dot1D(t->weights,t->current_node->weights->p,t->weights, t->weights_size);
            }
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_SUM){// sum weights
            if(t->current_node->weights->size == 1){// by single value
                add_value(t->weights, t->current_node->weights->single_p, t->weights, t->weights_size);
            } 
            else{// by same matrix dimension
                sum1D(t->weights,t->current_node->weights->p,t->weights, t->weights_size);
            }
        }
        
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MATRIX_MUL){// sum weights
            int size1 = t->weights_size/t->current_node->weights->dimension1;
            int size2 = t->current_node->weights->dimension1;
            int size3 = t->current_node->weights->dimension2;
            float* new_weights = (float*)calloc(size1*size3,sizeof(float));
            mat_mul(t->weights,t->current_node->weights->p, new_weights, size1,size2,size3);
            free(t->weights);
            t->weights = new_weights;
            t->weights_size = size1*size3;
        }
        
    }
    
    else if(node_state(t->current_node) == ACO_IS_BIAS){// operations about biases [copy, mul, add]
        if(t->edge_taken->operation_flag == ACO_OPERATION_COPY){// copying biases
            free(t->biases);
            t->biases = (float*)calloc(t->current_node->biases->size,sizeof(float));
            copy_array(t->current_node->biases->p,t->biases,t->current_node->biases->size);
            t->biases_size = t->current_node->biases->size;
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_MUL){// multiply biases
            if(t->current_node->biases->size == 1){// by single value
                mul_value(t->biases, t->current_node->biases->single_p, t->biases, t->biases_size);
            } 
            else{// by same matrix dimension
                dot1D(t->biases,t->current_node->biases->p,t->biases, t->biases_size);
            }
        }
        else if(t->edge_taken->operation_flag == ACO_OPERATION_SUM){// sum biases
            if(t->current_node->biases->size == 1){// by single value
                add_value(t->biases, t->current_node->biases->single_p, t->biases, t->biases_size);
            } 
            else{// by same matrix dimension
                sum1D(t->biases,t->current_node->biases->p,t->biases, t->biases_size);
            }
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_ACTIVATION){// just execution
        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            t->fcls[t->n_fcl-1]->activation_flag = t->current_node->a->activation_flag;
        }
    }
    
    else if(node_state(t->current_node) == ACO_IS_FCL){// just execution

        if(t->edge_taken->operation_flag == ACO_OPERATION_EXECUTE_NODE){
            if(t->fcls == NULL){
                t->fcls = (fcl**)malloc(sizeof(fcl*));
                t->fcls[0] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[0]);
                free(t->fcls[0]->weights);
                free(t->fcls[0]->biases);
                t->fcls[0]->weights = t->weights;
                t->fcls[0]->biases = t->biases;
            }
            else{
                t->fcls = (fcl**)realloc(t->fcls, sizeof(fcl*)*(t->n_fcl+1));
                t->fcls[t->n_fcl] = fully_connected(t->current_node->f->input_size,t->current_node->f->output_size, t->layers,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
                make_the_fcl_only_for_ff(t->fcls[t->n_fcl]);
                free(t->fcls[t->n_fcl]->weights);
                free(t->fcls[t->n_fcl]->biases);
                t->fcls[t->n_fcl]->weights = t->weights;
                t->fcls[t->n_fcl]->biases = t->biases;
            }
            t->weights = NULL;
            t->biases = NULL;
            t->weights_size = 0;
            t->biases_size = 0;
            t->n_fcl++;
            t->layers++;
        }
    }
}

// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_next(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = -1,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    float* pheromone = (float*)calloc(n_edges,sizeof(float));
    float* prob = (float*)calloc(n_edges,sizeof(float));
    int* indices = (int*)calloc(n_edges,sizeof(int));
    for(i = 0; i < n_edges; i++){
        pheromone[i] = (float)(es[i]->pheromone);
        indices[i] = i;
    }
    compute_prob_average(pheromone, prob, n_edges);
    sort(prob,indices,0,n_edges-1);
    float r = r2();
    for(i = 0; i < n_edges; i++){
        if(r <= prob[indices[i]]){
            t->edge_taken = es[indices[i]];
            t->current_node = t->edge_taken->output;
            ret = indices[i];
            break;
        }
        r-=prob[indices[i]];
    }
    if(i == n_edges){
        t->edge_taken = es[indices[n_edges-1]];
        t->current_node = t->edge_taken->output;
        ret = indices[indices[n_edges-1]];
    }
    t->edge_taken->flag = 1;
    t->current_node->flag = 1;
    free(pheromone);
    free(prob);
    free(indices);    
    return ret;
}
// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_next_weithout_setting_flags(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = -1,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    double* pheromone = (double*)calloc(n_edges,sizeof(double));
    float* prob = (float*)calloc(n_edges,sizeof(float));
    int* indices = (int*)calloc(n_edges,sizeof(int));
    for(i = 0; i < n_edges; i++){
        pheromone[i] = (double)(es[i]->pheromone);
        indices[i] = i;
    }
    compute_prob_average_double(pheromone, prob, n_edges);
    sort(prob,indices,0,n_edges-1);
    float r = r2();
    for(i = 0; i < n_edges; i++){
        if(r <= prob[indices[i]]){
            t->edge_taken = es[indices[i]];
            t->current_node = t->edge_taken->output;
            ret = indices[i];
            break;
        }
        r-=prob[indices[i]];
    }
    if(i == n_edges){
        t->edge_taken = es[indices[n_edges-1]];
        t->current_node = t->edge_taken->output;
        ret = indices[n_edges-1];
    }
    free(pheromone);
    free(prob);
    free(indices);    
    return ret;
}
// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_next_weithout_setting_flags_according_to_nodes(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = -1,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    double* pheromone = (double*)calloc(n_edges,sizeof(double));
    float* prob = (float*)calloc(n_edges,sizeof(float));
    int* indices = (int*)calloc(n_edges,sizeof(int));
    for(i = 0; i < n_edges; i++){
        pheromone[i] = (double)(es[i]->output->pheromone);
        indices[i] = i;
    }
    compute_prob_average_double(pheromone, prob, n_edges);
    sort(prob,indices,0,n_edges-1);
    float rr = r2();
    float r = r2();
    /* with levy fligth
    if(rr > t->levy_threshold){
        float s = 1.0/t->levy_ratio * (1.0-t->levy_threshold)/(1.0-rr);
        if (s > 1)
            s = 1;
        r = 1.0 - (1.0-r)*(1.0/s);
    }*/
    for(i = 0; i < n_edges; i++){
        if(r <= prob[indices[i]]){
            t->edge_taken = es[indices[i]];
            t->current_node = t->edge_taken->output;
            ret = indices[i];
            break;
        }
        r-=prob[indices[i]];
    }
    if(i == n_edges){
        t->edge_taken = es[indices[n_edges-1]];
        t->current_node = t->edge_taken->output;
        ret = indices[n_edges-1];
    }
    free(pheromone);
    free(prob);
    free(indices);    
    return ret;
}
// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_next_weithout_setting_flags_according_to_best(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    
    int ret = -1,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    double pheromone = es[0]->output->pheromone;
    int best_index = 0;
    for(i = 1; i < n_edges; i++){
        //printf("%d, %d\n",i,t->current_node->n_outputs);
        if((double)(es[i]->output->pheromone) > pheromone){
            pheromone = (double)(es[i]->output->pheromone);
            best_index = i;
        }
    }
    t->edge_taken = es[best_index];
    t->current_node = t->edge_taken->output;
    ret = best_index;
    return ret;
}

// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_next_by_index(aco_tracker* t, int index){
    if(t->current_node->n_outputs == 0){
        printf("returns -1\n");
        return -1;
    }
    int ret = -1,i;
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[index];
    t->current_node = t->edge_taken->output;
    //t->edge_taken->flag = 1;
    //t->current_node->flag = 1;
    return index;
}

// returns 1 if other next nodes exists, 0 otherwise
int update_aco_tracker_next_according_to_path(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min){
    if(!path_size)
        return -1;
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    
    int ret = path[path_index];
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[ret];
    t->edge_taken->pheromone = t->edge_taken->pheromone*p + pheromone;
    if(t->edge_taken->pheromone > t_max)
        t->edge_taken->pheromone = t_max;
    if(t->edge_taken->pheromone < t_min)
        t->edge_taken->pheromone = t_min;
    t->edge_taken->flag = 1;
    t->current_node = t->edge_taken->output;
    update_aco_tracker_next_according_to_path(t,pheromone,path,path_index+1,path_size-1,p,t_max,t_min);
    return ret;
}
// returns 1 if other next nodes exists, 0 otherwise
int update_aco_tracker_next_according_to_path_according_to_nodes(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min){
    if(!path_size)
        return -1;
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    
    int ret = path[path_index];
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[ret];
    t->edge_taken->output->pheromone = t->edge_taken->output->pheromone*p + pheromone;
    if(t->edge_taken->output->pheromone > t_max)
        t->edge_taken->output->pheromone = t_max;
    if(t->edge_taken->output->pheromone < t_min)
        t->edge_taken->output->pheromone = t_min;
    t->edge_taken->output->flag = 1;
    t->current_node = t->edge_taken->output;
    update_aco_tracker_next_according_to_path_according_to_nodes(t,pheromone,path,path_index+1,path_size-1,p,t_max,t_min);
    return ret;
}
// returns 1 if other next nodes exists, 0 otherwise
int update_aco_tracker_next_according_to_path_according_to_subnodes(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min, int init_node, int final_node){
    if(!path_size)
        return -1;
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    
    int ret = path[path_index];
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[ret];
    if(path_index >= init_node && path_index < final_node){
        t->edge_taken->output->pheromone = t->edge_taken->output->pheromone*p + pheromone;
        if(t->edge_taken->output->pheromone > t_max)
            t->edge_taken->output->pheromone = t_max;
        if(t->edge_taken->output->pheromone < t_min)
            t->edge_taken->output->pheromone = t_min;
        t->edge_taken->output->flag = 1;
    }
    t->current_node = t->edge_taken->output;
    update_aco_tracker_next_according_to_path_according_to_subnodes(t,pheromone,path,path_index+1,path_size-1,p,t_max,t_min, init_node, final_node);
    return ret;
}

// returns 1 if other next nodes exists, 0 otherwise
int update_aco_tracker_next_according_to_path_no_min_max(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min){
    if(!path_size)
        return -1;
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = path[path_index];
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[ret];
    if(!t->edge_taken->flag)
        t->edge_taken->pheromone*=p;
    t->edge_taken->pheromone += pheromone;
    t->edge_taken->flag = 1;
    t->current_node = t->edge_taken->output;
    update_aco_tracker_next_according_to_path_no_min_max(t,pheromone,path,path_index+1,path_size-1,p,t_max,t_min);
    return ret;
}

// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_best_next(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = 0,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    double pheromone = es[0]->pheromone;
    for(i = 1; i < n_edges; i++){
        if(es[i]->pheromone > pheromone){
            pheromone = es[i]->pheromone;
            ret = i;
        }
    }
    t->edge_taken = es[ret];
    t->current_node = t->edge_taken->output;
    return ret;
}

// returns 1 if other next nodes exists, 0 otherwise
int aco_tracker_best_next_set_flag(aco_tracker* t){
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = 0,i;
    aco_edge** es = t->current_node->outputs;
    int n_edges = t->current_node->n_outputs;
    double pheromone = es[0]->pheromone;
    for(i = 1; i < n_edges; i++){
        if(es[i]->pheromone > pheromone){
            pheromone = es[i]->pheromone;
            ret = i;
        }
    }
    t->current_node->flag = 1;
    t->edge_taken = es[ret];
    t->current_node = t->edge_taken->output;
    return ret;
}

// returns 1 if other next nodes exists, 0 otherwise
int update_aco_tracker_next_according_to_path_best(aco_tracker* t, int* path, int path_index, int path_size){
    if(!path_size)
        return -1;
    if(t->current_node->n_outputs == 0){
        return -1;
    }
    int ret = path[path_index];
    aco_edge** es = t->current_node->outputs;
    t->edge_taken = es[ret];
    t->current_node = t->edge_taken->output;
    t->current_node->flag = 1;
    update_aco_tracker_next_according_to_path_best(t,path,path_index+1,path_size-1);
    return ret;
}


void sign_best(aco_struct* s){
    aco_tracker* t = init_aco_tracker();
    t->current_node = s->root;
    int ret = 0;
    update_aco_tracker_next_according_to_path_best(t, s->best_trail, 0, s->length_best_trail);

}

model* build_model_from_tracker(aco_tracker* t){
    t->m = network(t->layers,0,t->n_cl, t->n_fcl,NULL,t->cls,t->fcls);  
    return t->m;
}

void reset_flags(aco_node* root){
    if(root == NULL)
        return;
    if(!root->flag)
        return;
    int i;
    root->flag = 0;
    for(i = 0; i < root->n_outputs; i++){
        reset_flags(root->outputs[i]->output);
        root->outputs[i]->flag = 0;
    }
    return;
}

void get_all_nodes_and_edges(aco_node* root, aco_node*** nodes, aco_edge*** edges, int* n_nodes, int* n_edges){
    if(root == NULL){
        return;
    }
    if(root->flag == 1){
        return;
    }
    int i;
    root->flag = 1;
    (*n_nodes)++;
    nodes[0] = realloc(nodes[0],sizeof(aco_node**)*(*n_nodes));
    nodes[0][(*n_nodes)-1] = root;
    for(i = 0; i < root->n_outputs; i++){
        (*n_edges)++;
        edges[0] = realloc(edges[0],sizeof(aco_edge**)*(*n_edges));
        edges[0][(*n_edges)-1] = root->outputs[i];
        get_all_nodes_and_edges(root->outputs[i]->output, nodes, edges, n_nodes, n_edges);
    }
    return;
}

void compute_all_nodes_edges(aco_node* root, aco_node*** nodes, aco_edge*** edges, int* n_nodes, int* n_edges){
    get_all_nodes_and_edges(root, nodes, edges, n_nodes, n_edges);
    int i;
    for(i = 0; i < (*n_nodes); i++){
        nodes[0][i]->flag = 0;
    }
    for(i = 0; i < (*n_edges); i++){
        edges[0][i]->flag = 0;
    }
}
