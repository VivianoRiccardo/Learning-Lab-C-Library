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

mcts_node* init_mcts_node(float* state, float q_value, float reward, float v, uint64_t state_size, uint64_t depth, uint64_t visit_count, uint64_t n_edges, uint64_t lstm_layers, uint64_t* h_states_size){
    mcts_node* n = (mcts_node*)malloc(sizeof(mcts_node));
    n->state = state;
    n->reward = reward;
    n->prefix_value_reward = 0;
    n->q_value = q_value;
    n->v = v;
    n->v_pow = 1;
    n->state_with_actions = NULL;
    n->state_size = state_size;
    n->depth = depth;
    n->visit_count = visit_count;
    n->n_edges = n_edges;
    n->edges = NULL;
    n->lstm_layers = lstm_layers;
    
    if(lstm_layers){
        n->hidden_states = (float**)malloc(sizeof(float*)*lstm_layers);
        n->cell_states = (float**)malloc(sizeof(float*)*lstm_layers);
        int i;
        for(i = 0; i < lstm_layers; i++){
            n->hidden_states[i] = (float*)calloc(h_states_size[i],sizeof(float));
            n->cell_states[i] = (float*)calloc(h_states_size[i],sizeof(float));
        }
    }
    else{
        n->hidden_states = NULL;
        n->cell_states = NULL;
    }
    free(h_states_size);
    n->h_states_size = NULL;
    if(n->n_edges){
        int i;
        n->edges = (mcts_edge**)malloc(sizeof(mcts_edge*)*n->n_edges);
        for(i = 0; i < n->n_edges; i++){
            n->edges[i] = NULL;
        }
    }
    return n;
}


void free_mcts_node(mcts_node* n){
    free_matrix((void**)n->hidden_states,n->lstm_layers);
    free_matrix((void**)n->cell_states,n->lstm_layers);
    free(n->state_with_actions);
    free(n->h_states_size);
    free(n->state);
    free(n->edges);
    free(n);
}

void mcts_node_set_state(mcts_node* n, float* state, uint64_t state_size){
    free(n->state);
    n->state = state;
    n->state_size = state_size;
}

int node_is_full_of_edges(mcts_node* n){
    int i;
    for(i = 0; i < n->n_edges; i++){
        if(n->edges[i] == NULL)
            return 0;
    }
    return 1;
}

void mcts_node_add_edge(mcts_node* n, mcts_edge* e){
    if(n->edges == NULL){
        n->n_edges++;
        n->edges = (mcts_edge**)malloc(sizeof(mcts_edge*)*n->n_edges);
        n->edges[0] = e;
        return;
    }
    
    if(n->edges[n->n_edges-1] != NULL){
        n->n_edges++;
        n->edges = (mcts_edge**)realloc(n->edges,sizeof(mcts_edge*)*n->n_edges);
        n->edges[n->n_edges-1] = e;
        return;
    }
    
    
    int i;
    for(i = 0; i < n->n_edges; i++){
        if(n->edges[i] == NULL){
            n->edges[i] = e;
            return;
        }
    }
}

mcts_edge* init_mcts_edge(mcts_node* input_node, mcts_node* output_node, float prior_probability){
    mcts_edge* e = (mcts_edge*)malloc(sizeof(mcts_edge));
    e->input_node = input_node;
    e->output_node = output_node;
    e->prior_probability = prior_probability;
    return e;
}

mcts* init_mcts(efficientzeromodel* m, float* init_state, double value_offset, double reward_offset, double gamma_reward, double dirichlet_alpha, double c_init, double c_base, double noise_epsilon, uint64_t init_state_size, uint64_t maximum_depth){
    mcts* t = (mcts*)malloc(sizeof(mcts));
    t->m = m;
    uint64_t* h_states_sizes = (uint64_t*)calloc(m->reward_prediction_rmodel->layers,sizeof(uint64_t));
    int i;
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        h_states_sizes[i] = m->reward_prediction_rmodel->lstms[i]->output_size;
    }
    mcts_node* root = init_mcts_node(init_state, 0, 0, 0, init_state_size,0,0,m->prediction_f_policy->output_dimension,m->reward_prediction_rmodel->layers,h_states_sizes);
    t->root = root;
    t->q_max = -1000000000;
    t->q_min = 1000000000;
    t->q_difference = 1000000000;
    t->c_init = c_init;
    t->c_base = c_base;
    t->noise_epsilon = noise_epsilon;
    t->n_nodes = 1;
    t->n_edges = 0;
    t->maximum_depth = maximum_depth;
    t->dirichlet_alpha = dirichlet_alpha;
    t->gamma_reward = gamma_reward;
    t->reward_offset = reward_offset;
    t->edges = NULL;
    t->nodes = (mcts_node**)malloc(sizeof(mcts_node*));
    
    

    
    if(t->m->prediction_f_policy->output_dimension){
        efficientzero_ff_prediction_f(t->m, init_state);
        t->edges = (mcts_edge**)malloc(sizeof(mcts_edge*)*t->m->prediction_f_policy->output_dimension);
        for(i = 0; i < t->m->prediction_f_policy->output_dimension; i++){
            t->root->edges[i] = init_mcts_edge(t->root,NULL,t->m->prediction_f_policy->output_layer[i]);
            t->edges[i] = t->root->edges[i];
        }
        t->n_edges = t->m->prediction_f_policy->output_dimension;
        efficientzero_reset_only_for_ff_prediction_f(t->m);
    }
    
    t->nodes[0] = root;
    t->value_offset = value_offset;
    return t;
}

void free_mcts(mcts* t){
    int i;
    for(i = 0; i < t->n_edges; i++){
        free(t->edges[i]);
    }
    for(i = 0; i < t->n_nodes; i++){
        free_mcts_node(t->nodes[i]);
    }
    free(t->edges);
    free(t->nodes);
    free(t);
}

double get_sum_visited_children_q_normalized_mcts_node(mcts* t, mcts_node* n){
    double sum;
    int i;
    for(sum = 0, i = 0; i < n->n_edges ; i++){
        if(n->edges[i] != NULL && n->edges[i]->output_node != NULL && n->edges[i]->output_node->visit_count > 0 && t->q_difference != 0.0)
            sum+=(n->edges[i]->output_node->q_value-t->q_min)/t->q_difference;
    }
    return sum;
}

double get_visited_children_q_mcts_node(mcts_node* n){
    double sum;
    int i;
    for(sum = 0, i = 0; i < n->n_edges ; i++){
        if(n->edges[i] != NULL && n->edges[i]->output_node != NULL && n->edges[i]->output_node->visit_count > 0)
            sum++;
    }
    return sum;
}


double visit_node(mcts* t, mcts_node* n, double q_hat_parent, uint64_t depth){
    
    if(depth >= t->maximum_depth){
        t->returned_node = n;
        n->q_value=(n->q_value*n->visit_count + n->reward + n->v)/((double)(n->visit_count+1));
        n->visit_count++;
        
        if(n->q_value < t->q_min)
            t->q_min = n->q_value;
        if(n->q_value > t->q_max)
            t->q_max = n->q_value;
        
        
        return n->prefix_value_reward;
        
    }
    
    int i, index = -1;
    double q_hat = (get_sum_visited_children_q_normalized_mcts_node(t, n) + q_hat_parent)/get_visited_children_q_mcts_node(n);
    
    if(!depth){// root
        double* p = (double*)calloc(n->n_edges,sizeof(double));//alloc probability
        int* indices = (int*)calloc(n->n_edges,sizeof(int));//alloc indices
        dirichlet_sample(t->dirichlet_alpha,n->n_edges,p);// dirichlet distribution
        for(i = 0; i < n->n_edges; i++){
            indices[i] = i;
        }
        // modify according to mcts root decision distribution
        for(i = 0; i < n->n_edges; i++){
            p[i] = p[i]*t->noise_epsilon + (1-t->noise_epsilon)*n->edges[i]->prior_probability;
        }
        // sort the probabilities
        sort_double(p,indices,0,n->n_edges-1);
        
        
        // select the right probability
        
        float random = r2();
        for(i = 0; i < n->n_edges; random -= p[indices[i]], i++){
            if(p[indices[i]] <= random){
                index = indices[i];
                break;
            }
        }
        
        free(p);// free
        free(indices);// free
        
    }
    
    else{
        index = 0;
        double ucb = 0;
        double ucb_temp = 0;
        if(n->edges[index]->output_node == NULL)
            ucb = q_hat;
        else
            ucb = n->edges[index]->output_node->q_value + n->edges[index]->prior_probability*(log((1.0+n->edges[index]->output_node->visit_count+t->c_base)/t->c_base) + t->c_init)*sqrtf(n->edges[index]->output_node->visit_count)/(1.0+n->edges[index]->output_node->visit_count);
        for(i = 1; i < n->n_edges; i++){
            if(n->edges[i]->output_node == NULL)
                ucb_temp = q_hat;
            else
                ucb_temp = (n->edges[i]->output_node->q_value-t->q_min)/(t->q_max-t->q_min) + n->edges[i]->prior_probability*(log((1.0+n->edges[i]->output_node->visit_count+t->c_base)/t->c_base) + t->c_init)*sqrtf(n->edges[i]->output_node->visit_count)/(1.0+n->edges[i]->output_node->visit_count);
            if(ucb_temp > ucb){
                ucb = ucb_temp;
                index = i;
            }
        }
    }
        
    // generate new node
    if(n->edges[index]->output_node == NULL){
        
        // set the action to the previous state
        set_vector_with_value(((float)(index+1))/((float)(n->n_edges)), n->state+t->m->dynamics_g->output_dimension,get_input_layer_size(t->m->dynamics_g)-t->m->dynamics_g->output_dimension);

        
        // create the state array that can contains also the actions
        float* new_state = (float*)calloc(get_input_layer_size(t->m->dynamics_g), sizeof(float));
        // output the state
        efficientzero_ff_dynamics_g(t->m, n->state);
        // copy it
        copy_array(t->m->dynamics_g->output_layer,new_state,t->m->dynamics_g->output_dimension);
        // reset the dynamic network that outputed the new state
        efficientzero_reset_only_for_ff_dynamics_g(t->m);
        
        // create the node
        uint64_t* h_states_sizes = (uint64_t*)calloc(t->m->reward_prediction_rmodel->layers,sizeof(uint64_t));
        int i;
        for(i = 0; i < t->m->reward_prediction_rmodel->layers; i++){
            h_states_sizes[i] = t->m->reward_prediction_rmodel->lstms[i]->output_size;
        }
        
        
        mcts_node* new_node = init_mcts_node(new_state, 0, 0, 0, n->state_size,depth+1,0,t->m->prediction_f_policy->output_dimension,t->m->reward_prediction_rmodel->layers,h_states_sizes);
        
        // set the node in the tree
        t->nodes = (mcts_node**)realloc(t->nodes,sizeof(mcts_node**)*(t->n_nodes+1));
        t->nodes[t->n_nodes] = new_node;
        t->n_nodes++;
        
        
        // get the policy and the value from the new state
        efficientzero_ff_prediction_f(t->m, new_state);
        
        t->edges = (mcts_edge**)realloc(t->edges,sizeof(mcts_edge**)*(t->n_edges+t->m->prediction_f_policy->output_dimension));
        for(i = 0; i < n->n_edges; i++){
            new_node->edges[i] = init_mcts_edge(new_node,NULL,t->m->prediction_f_policy->output_layer[i]);
            t->edges[t->n_edges+i] = new_node->edges[i];
        }
        t->n_edges+=t->m->prediction_f_policy->output_dimension;
        
        efficientzero_ff_reward_prediction_single_cell(t->m, new_state, n->hidden_states, n->cell_states, new_node->hidden_states, new_node->cell_states, (depth + 1)%t->m->lstm_window);
        
        // get the value-prefix
        
        float reward = mul_array_values_by_indices(t->m->reward_prediction_temporal_model->output_layer, t->reward_offset, t->m->reward_prediction_temporal_model->output_dimension);
        
        // get the v
        
        float value = mul_array_values_by_indices(t->m->prediction_f_value->output_layer, t->value_offset, t->m->prediction_f_value->output_dimension);
        
        float current_node_reward = reward;
        
        if (depth%t->m->lstm_window){
            current_node_reward -= n->prefix_value_reward;
            // why this? the lstm cell predict the value-prefix in a lstm window, 
            // so if this new node is not a node where the h_previous has been reset than the reward predicted
            // is the prefix from the last cell where h previous has been reset up to this.
            // so since to update the q value of the node we need just the value-prefix of future rewards we must remove
            // from the current reward the values of previous rewards
        }
        
        // set the reward
        new_node->reward = current_node_reward;//reward for this state 
        new_node->prefix_value_reward = reward;// prefix value reward
        new_node->v = t->gamma_reward*value;// v * gamma
        // set the visit count
        new_node->visit_count = 1;
        // set the q value
        new_node->q_value = new_node->reward + new_node->v;
        
        efficientzero_reset_only_for_ff_prediction_f(t->m);
        
        if(new_node->q_value < t->q_min)
            t->q_min = new_node->q_value;
        if(new_node->q_value > t->q_max)
            t->q_max = new_node->q_value;
            
        efficientzero_reset_only_for_ff_reward_prediction_single_cell(t->m);
        n->edges[index]->output_node = new_node;
        t->returned_node = new_node;
        
        
        // now update the current node
        
        return t->returned_node->v;
    }
    
    else{
        double v_next = visit_node(t, n->edges[index]->output_node, q_hat, depth+1);
        
        if(n->edges[index]->output_node == t->returned_node){
            return v_next;
        }
        
        double reward = t->returned_node->prefix_value_reward;
        
        int ret_node_ind = t->returned_node->depth%t->m->lstm_window;
        int child_node_ind = n->edges[index]->output_node->depth%t->m->lstm_window;
        
        if(!child_node_ind || child_node_ind >= ret_node_ind || t->returned_node->depth - n->edges[index]->output_node->depth > t->m->lstm_window){
            n->q_value = (n->edges[index]->output_node->visit_count*n->edges[index]->output_node->q_value + (reward+v_next))/((double)(n->edges[index]->output_node->visit_count+1));
        }
        
        else{
            v_next*=t->gamma_reward;
            n->q_value = (n->edges[index]->output_node->visit_count*n->q_value + (reward - n->edges[index]->output_node->prefix_value_reward+v_next))/((double)(n->edges[index]->output_node->visit_count+1));
        }
        
        
        n->edges[index]->output_node->visit_count++;
        if(n->edges[index]->output_node->q_value < t->q_min)
            t->q_min = n->edges[index]->output_node->q_value;
        if(n->edges[index]->output_node->q_value > t->q_max)
            t->q_max = n->edges[index]->output_node->q_value;
        return v_next;
    }
        
}


float* get_mcts_probability(mcts* t, float temperature){
    int i;
    double sum =0;
    float* p = (float*)calloc(t->m->prediction_f_policy->output_dimension,sizeof(float));
    for(i = 0; i < t->m->prediction_f_policy->output_dimension; i++){
        if(t->root->edges[i] != NULL && t->root->edges[i]->output_node != NULL){
            t->root->edges[i]->output_node->visit_count = pow(t->root->edges[i]->output_node->visit_count,1.0/temperature);
            sum+=t->root->edges[i]->output_node->visit_count;
        }
    }
    
    for(i = 0; i < t->m->prediction_f_policy->output_dimension; i++){
        if(t->root->edges[i] != NULL && t->root->edges[i]->output_node != NULL){
            p[i] = ((double)t->root->edges[i]->output_node->visit_count)/sum;
        }
    }
    
    return p;
}

double get_mcts_v(mcts* t, float* p){
    int i;
    double v = 0;
    for(i = 0; i < t->m->prediction_f_policy->output_dimension; i++){
        if(t->root->edges[i] != NULL && t->root->edges[i]->output_node != NULL){
            v += p[i]*(t->root->edges[i]->output_node->q_value - t->q_min)/(t->q_max-t->q_min);
        }
    }
    return v;
}



