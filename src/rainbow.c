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

rainbow* init_rainbow(int sampling_flag, int gd_flag, int lr_decay_flag, int feed_forward_flag, int training_mode, int clipping_flag, int adaptive_clipping_flag, int batch_size,int threads, 
                      uint64_t diversity_driven_q_functions, uint64_t epochs_to_copy_target, uint64_t max_buffer_size, uint64_t n_step_rewards, uint64_t stop_epsilon_greedy, uint64_t past_errors, uint64_t lr_epoch_threshold,
                      float max_epsilon, float min_epsilon, float epsilon_decay, float epsilon, float alpha_priorization, float beta_priorization, float lambda_value,float gamma, float tau_copying, float beta1, float beta2,
                      float beta3, float k_percentage, float clipping_gradient_value, float adaptive_clipping_gradient_value, float lr, float lr_minimum, float lr_maximum, float lr_decay, float momentum,
                      float diversity_driven_threshold, float diversity_driven_decay, float diversity_driven_minimum, float diversity_driven_maximum, float beta_priorization_increase,
                      dueling_categorical_dqn* online_net, dueling_categorical_dqn* target_net, dueling_categorical_dqn** online_net_wlp,
                      dueling_categorical_dqn** target_net_wlp){
                          
    
    if(sampling_flag != UNIFORM_SAMPLING && sampling_flag != REWARD_SAMPLING && sampling_flag != RANKED_SAMPLING){
        fprintf(stderr,"Error: sampling flag not recognized\n");
        exit(1);
    }
    
    
    if(beta_priorization_increase < 0 || beta_priorization_increase > 1){
        fprintf(stderr,"Error: beta_priorization_increase should be in range [0,1]\n"),
        exit(1);
    }
    if(diversity_driven_decay < 0 || diversity_driven_decay > 1){
        fprintf(stderr,"Error: diversity driven decay should be in range [0,1]\n"),
        exit(1);
    }
    
    if(diversity_driven_minimum < 0 || diversity_driven_minimum > 1 || diversity_driven_minimum > diversity_driven_maximum){
        fprintf(stderr,"Error: diversity driven minimum should be in range [0,1]\n"),
        exit(1);
    }
    
    if(diversity_driven_maximum < 0 || diversity_driven_maximum > 1){
        fprintf(stderr,"Error: diversity driven maximum should be in range [0,1]\n"),
        exit(1);
    }
    
    if(lr_decay_flag != LR_NO_DECAY && lr_decay_flag != LR_ANNEALING_DECAY && lr_decay_flag != LR_CONSTANT_DECAY && lr_decay_flag != LR_STEP_DECAY && lr_decay_flag != LR_TIME_BASED_DECAY){
        fprintf(stderr, "Error: no error decay flag recognized\n");
        exit(1);
    }
    
    if(feed_forward_flag != FULLY_FEED_FORWARD && feed_forward_flag != EDGE_POPUP){
        fprintf(stderr,"Error: feed forward flag not recognized\n");
        exit(1);
    }
    
    if(training_mode != GRADIENT_DESCENT && training_mode != EDGE_POPUP){
        fprintf(stderr,"Error: training mode not recognized\n");
        exit(1);
    }
    
    if(gd_flag != NESTEROV && gd_flag != ADAM && gd_flag != ADAMOD && gd_flag != RADAM && gd_flag != DIFF_GRAD){
        fprintf(stderr,"Error: gradient descent flag not recognized\n");
        exit(1);
    }
    
    if(batch_size <= 0){
        fprintf(stderr,"Error: batch size can't be <= 0\n");
        exit(1);
    }
    
    if(threads <= 0){
        fprintf(stderr,"Error: threads can't be <= 0\n");
        exit(1);
    }
    
    if(threads > batch_size){
        fprintf(stderr,"Error: threads should be <= batch size\n");
        exit(1);
    }
    
    if(max_buffer_size<=1){
        fprintf(stderr,"Error: buffer size can't be <= 1\n");
        exit(1);
    }
    
    if(!n_step_rewards){
        fprintf(stderr,"Error: n step rewards can't be = 0\n");
        exit(1);
    }
    
    if(max_epsilon > 1){
        fprintf(stderr,"Error: max epsilon can't be > 1\n");
        exit(1);
    }
    
    if(min_epsilon <= 0){
        fprintf(stderr,"Error: min epsilon can't be <= 0\n");
        exit(1);
    }
    
    if(min_epsilon > max_epsilon){
        fprintf(stderr,"Error: min epsilond can't me > max epsilon\n");
        exit(1);
    }
    
    if(epsilon_decay < 0){
        fprintf(stderr,"Error: epsilon decay can't be < 0\n");
        exit(1);
    }
    
    if(epsilon < 0 || epsilon > 1){
        fprintf(stderr,"Error: epsilon range in [0,1]\n");
        exit(1);
    }
    
    
    if(diversity_driven_q_functions < batch_size){
        fprintf(stderr,"Error: diversity driven q functions can't be < batch_size\n");
        exit(1);
    }
    
    
    if(alpha_priorization < 0 || alpha_priorization > 1){
        fprintf(stderr,"Error: alpha_priorization range in [0,1]\n");
        exit(1);
    }
    
    if(beta_priorization < 0 || beta_priorization > 1){
        fprintf(stderr,"Error: beta_priorization range in [0,1]\n");
        exit(1);
    }
    if(lambda_value < 0 || lambda_value > 1){
        fprintf(stderr,"Error: lambda_value range in [0,1]\n");
        exit(1);
    }
    
    if(gamma < 0 || gamma > 1){
        fprintf(stderr,"Error: gamma range in [0,1]\n");
        exit(1);
    }
    
    if(tau_copying < 0 || tau_copying > 1){
        fprintf(stderr,"Error: tau_copying range in [0,1]\n");
        exit(1);
    }
    
    if(beta1 < 0 || beta1 > 1){
        fprintf(stderr,"Error: beta1 range in [0,1]\n");
        exit(1);
    }
    
    if(beta2 < 0 || beta2 > 1){
        fprintf(stderr,"Error: beta2 range in [0,1]\n");
        exit(1);
    }
    
    if(beta3 < 0 || beta3 > 1){
        fprintf(stderr,"Error: beta3 range in [0,1]\n");
        exit(1);
    }
    
    if(k_percentage < 0 || k_percentage > 1){
        fprintf(stderr,"Error: k_percentage range in [0,1]\n");
        exit(1);
    }
    
    if(clipping_gradient_value < 0){
        fprintf(stderr,"Error: clipping_gradient_value must be >= 0\n");
        exit(1);
    }
    
    if(adaptive_clipping_gradient_value < 0 || adaptive_clipping_gradient_value > 1){
        fprintf(stderr,"Error: adaptive_clipping_gradient_value range in [0,1]\n");
        exit(1);
    }
    if(lr < 0 || lr > 1){
        fprintf(stderr,"Error: lr range in [0,1]\n");
        exit(1);
    }
    if(lr_minimum < 0 || lr_minimum > 1){
        fprintf(stderr,"Error: lr_minimum range in [0,1]\n");
        exit(1);
    }
    
    if(lr_maximum < 0 || lr_maximum > 1){
        fprintf(stderr,"Error: lr_maximum range in [0,1]\n");
        exit(1);
    }
    if(lr_decay < 0 || lr_decay > 1){
        fprintf(stderr,"Error: lr_decay range in [0,1]\n");
        exit(1);
    }
    if(momentum < 0 || momentum > 1){
        fprintf(stderr,"Error: momentum range in [0,1]\n");
        exit(1);
    }
    
    if(lr_minimum > lr_maximum){
        fprintf(stderr,"Error: lr minimum can't be greater than lr maximum\n");
        exit(1);
    }
    
    if(online_net == NULL || online_net_wlp == NULL || target_net == NULL || target_net_wlp == NULL){
        fprintf(stderr,"Error: the nets u passed are set to null\n");
        exit(1);
    }
    
    if(diversity_driven_q_functions >= max_buffer_size){
        fprintf(stderr,"Error: diversity_driven_q_functions can't be >= max_buffer_size\n");
        exit(1);
    }
    
    if(!diversity_driven_q_functions){
        fprintf(stderr,"Error: diversity_driven_q_functions can't be < 1\n");
        exit(1);
    }
    
    if(diversity_driven_threshold<= 0){
        fprintf(stderr,"Error: threshold should be > 0\n"),
        exit(1);
    }
    rainbow* r = (rainbow*)malloc(sizeof(rainbow));
    r->max_epsilon = max_epsilon;// maximum epsilon that epsilon can reach
    r->min_epsilon = min_epsilon;// minimum epsilon that epsilon can reach
    r->epsilon_decay = epsilon_decay;// the epsilon decay parameter for the epsilon
    r->epsilon = epsilon;// the epsilon for the epsilon greedy exploration
    r->alpha_priorization = alpha_priorization;// alpha priorization usually set to 0.6
    r->beta_priorization = beta_priorization;// beta priorization to weight the loss coming from the priorization usually set to 0.4 and linear increasing
    r->lambda_value = lambda_value;// lambda value for the n step reward computation
    r->tau_copying = tau_copying;// the value that will copied into the target network from the online net
    r->beta1 = beta1;// the beta1 parameter
    r->beta2 = beta2;// the beta2 parameter
    r->beta3 = beta3;// the betamod parameter
    r->k_percentage = k_percentage;// k percentage for a possible edge popup training
    r->momentum = momentum;// momentum for gradient descent
    r->clipping_gradient_value = clipping_gradient_value;// clipping gradient value for the diversity driven
    r->adaptive_clipping_gradient_value = adaptive_clipping_gradient_value;// value for a possible adaptive clipping gradient
    r->lr = lr;// learning rate
    r->initial_lr = lr;// initial learning rate
    r->lr_minimum = lr_minimum;// minimum learning rate
    r->lr_maximum = lr_maximum;//  maximum learning rate
    r->feed_forward_flag = feed_forward_flag;// feed forward flag across the network
    r->gd_flag = gd_flag;// gradient descent flag for gradient descent
    r->training_mode = training_mode;// training mode for the whole network
    r->clipping_flag = clipping_flag;// clipping flag, useless for now
    r->adaptive_clipping_flag = adaptive_clipping_flag;// adaptive clipping flag, is set we can perform the adaptive clipping gradient
    r->batch_size = batch_size;// the batch size
    r->threads  = threads;// the number of threads
    r->epochs_to_copy_target = epochs_to_copy_target;//after each epochs to copy target we copy target net from online net
    r->lr_decay_flag = lr_decay_flag;// learning rate decay flag
    r->sum_error_priorization_buffer = 0;// the total error ranked priorization over the buffer
    r->positive_sum_error_priorization_buffer = 0;// the total error ranked priorization over the buffer
    r->negative_sum_error_priorization_buffer = 0;// the total error ranked priorization over the buffer
    r->neutral_sum_error_priorization_buffer = 0;// the total error ranked priorization over the buffer
    r->action_taken_iteration = 0;// the iteration of the action taken
    r->max_buffer_size = max_buffer_size;// the maximum size of the buffer where we store transitions
    r->train_iteration = 1;// the iteration of the update
    r->buffer_current_index = 0;// where we actually will store the next state
    r->n_step_rewards = n_step_rewards;// n step reward for the td error
    r->stop_epsilon_greedy = stop_epsilon_greedy;// when we reached stop epsilon greedy iteration from action taken iteration we don't use epsilon greedy anymore
    r->lr_epoch_threshold = lr_epoch_threshold;//the learning rate epoch threshold to update the learning rate
    r->buffer_state_t = (float**)calloc(r->max_buffer_size,sizeof(float*));// allocation the buffer of states t
    r->buffer_state_t_1 = (float**)calloc(r->max_buffer_size,sizeof(float*));// allocation the buffer of states t + 1
    r->nonterminal_state_t_1 = (int*)calloc(r->max_buffer_size,sizeof(int));// allocation tells us if state t+1 is terminal or not
    r->actions = (int*)calloc(r->max_buffer_size,sizeof(int));// allocation buffer of actions
    r->rewards = (float*)calloc(r->max_buffer_size,sizeof(float));// allocation buffer of rewards
    r->online_net = online_net;// the online net
    r->online_net_wlp = online_net_wlp;// the online net without learning parameters
    r->target_net = target_net;// the target net
    r->target_net_wlp = target_net_wlp;// the target net without learning parameters
    r->diversity_driven_q_functions = diversity_driven_q_functions;// number of the last recorded q functions from our net or other nets
    r->diversity_driven_q_functions_counter = 0;// the index where we will store the next q function
    r->diversity_driven_q_functions_buffer = (float*)calloc(diversity_driven_q_functions*online_net->action_size,sizeof(float));// allocation the buffer of the last q functions
    r->diversity_driven_states = (float**)calloc(diversity_driven_q_functions,sizeof(float*));// allocation// the buffer of the last states of the last q functions
    
    if(sampling_flag == RANKED_SAMPLING){
        r->error_priorization = (float*)calloc(r->max_buffer_size,sizeof(float));// allocation where the td errors are stored
        r->error_indices = (int*)calloc(r->max_buffer_size,sizeof(int));// allocation the indices of the errors from the heap to the buffer of above, error_priorization[error_indices[i]]
        r->reverse_error_indices = (int*)calloc(r->max_buffer_size,sizeof(int));
        r->ranked_values = (float*)calloc(r->max_buffer_size,sizeof(float));// allocation buffer of ranked values. we use a heap, ranked values tells us the value of the rank of that ceil
        r->recursive_cumulative_ranked_values = (float*)calloc(r->max_buffer_size,sizeof(float));// allocation is a recursive tree where the parent of the heap contains as value the sum of the right and left sub trees
        r->positive_rewards = NULL;
        r->negative_rewards = NULL;
        r->neutral_rewards = NULL;
        r->positive_ranked_values = NULL;
        r->positive_recursive_cumulative_ranked_values = NULL;
        r->negative_ranked_values = NULL;
        r->negative_recursive_cumulative_ranked_values = NULL;
        r->neutral_ranked_values = NULL;
        r->neutral_recursive_cumulative_ranked_values = NULL;
        r->positive_reverse_indices = NULL;
        r->negative_reverse_indices = NULL;
        r->neutral_reverse_indices = NULL;
        r->positive_batch = NULL;
        r->negative_batch = NULL;
        r->neutral_batch = NULL;
        r->uniform_sampling_indices = NULL;
    }
    else if(sampling_flag == REWARD_SAMPLING){
        r->error_priorization =NULL;
        r->error_indices = NULL;
        r->reverse_error_indices = NULL;
        r->ranked_values = NULL;// allocation buffer of ranked values. we use a heap, ranked values tells us the value of the rank of that ceil
        r->recursive_cumulative_ranked_values = NULL;// allocation is a recursive tree where the parent of the heap contains as value the sum of the right and left sub trees
        r->positive_rewards = NULL;
        r->negative_rewards = NULL;
        r->neutral_rewards = NULL;
        r->positive_batch = (uint*)calloc(r->batch_size,sizeof(uint));
        r->negative_batch = (uint*)calloc(r->batch_size,sizeof(uint));
        r->neutral_batch = (uint*)calloc(r->batch_size,sizeof(uint));
        r->positive_ranked_values = (int*)calloc(r->max_buffer_size,sizeof(int));
        r->positive_recursive_cumulative_ranked_values = (float*)calloc(r->max_buffer_size,sizeof(float));
        r->negative_ranked_values = (int*)calloc(r->max_buffer_size,sizeof(int));
        r->negative_recursive_cumulative_ranked_values = (float*)calloc(r->max_buffer_size,sizeof(float));
        r->neutral_ranked_values = (int*)calloc(r->max_buffer_size,sizeof(int));
        r->neutral_recursive_cumulative_ranked_values = (float*)calloc(r->max_buffer_size,sizeof(float));
        r->positive_reverse_indices = NULL;
        r->negative_reverse_indices = NULL;
        r->neutral_reverse_indices = NULL;
        r->uniform_sampling_indices = NULL;
    }
    
    else if(sampling_flag == UNIFORM_SAMPLING){
        r->error_priorization =NULL;
        r->error_indices = NULL;
        r->ranked_values = NULL;// allocation buffer of ranked values. we use a heap, ranked values tells us the value of the rank of that ceil
        r->recursive_cumulative_ranked_values = NULL;// allocation is a recursive tree where the parent of the heap contains as value the sum of the right and left sub trees
        r->reverse_error_indices = NULL;
        r->positive_rewards = NULL;
        r->negative_rewards = NULL;
        r->neutral_rewards = NULL;
        r->positive_ranked_values = NULL;
        r->positive_recursive_cumulative_ranked_values = NULL;
        r->negative_ranked_values = NULL;
        r->negative_recursive_cumulative_ranked_values = NULL;
        r->neutral_ranked_values = NULL;
        r->neutral_recursive_cumulative_ranked_values = NULL;
        r->positive_reverse_indices = NULL;
        r->negative_reverse_indices = NULL;
        r->neutral_reverse_indices = NULL;
        r->positive_batch = NULL;
        r->negative_batch = NULL;
        r->neutral_batch = NULL;
        r->uniform_sampling_indices = (int*)calloc(r->max_buffer_size,sizeof(int));
    }
    
    r->diversity_driven_threshold = diversity_driven_threshold;// the threshold used to update alpha for the diversity driven exploration rule
    r->alpha = 0.1;// the initial alpha value

    r->last_errors_dqn = NULL;
    r->last_errors_diversity_driven = NULL;
    
    r->past_errors = past_errors;
    r->past_errors_counter = 0;
    r->beta_priorization_increase = beta_priorization_increase;
    r->diversity_driven_decay = diversity_driven_decay;
    r->diversity_driven_minimum = diversity_driven_minimum;
    r->diversity_driven_maximum = diversity_driven_maximum;
    //r->reverse_error_indices = NULL;
    
    
    // for training
    r->array_to_shuffle = (int*)calloc(r->diversity_driven_q_functions,sizeof(int));
    r->batch = (uint*)calloc(r->batch_size,sizeof(uint));
    r->reverse_batch = (uint*)calloc(r->batch_size,sizeof(uint));
    r->temp_states_t = (float**)calloc(r->batch_size,sizeof(float*));
    r->temp_states_t_1 = (float**)calloc(r->batch_size,sizeof(float*));
    r->temp_diversity_states_t = (float**)calloc(r->batch_size,sizeof(float*));
    r->qs = (float**)calloc(r->batch_size,sizeof(float*));
    r->temp_nonterminal_state_t_1 = (int*)calloc(r->batch_size,sizeof(int));
    r->temp_actions = (int*)calloc(r->batch_size,sizeof(int));
    r->temp_rewards = (float*)calloc(r->batch_size,sizeof(float));
    r->new_errors = (float*)calloc(r->batch_size,sizeof(float));
    r->weighted_errors = (float*)calloc(r->batch_size,sizeof(float));
    r->sampling_flag = sampling_flag;
    r->positive_rewards_counter = 0;
    r->positive_rewards_length = 0;
    r->negative_rewards_counter = 0;
    r->negative_rewards_length = 0;
    r->neutral_rewards_counter = 0;
    r->neutral_rewards_length = 0;
    int i;
    
    for(i = 0; i < r->diversity_driven_q_functions; i++){
        r->array_to_shuffle[i] = i;
    }
    
    r->gamma = 1;
    for(i = 0; i < n_step_rewards; i++){
        r->gamma*=lambda_value;
    }
    
    set_dueling_categorical_dqn_beta(r->online_net,beta1,beta2);
    set_dueling_categorical_dqn_beta_adamod(r->online_net,beta3);
    set_dueling_categorical_dqn_beta_adamod(r->target_net,beta3);
    set_dueling_categorical_dqn_beta(r->target_net,beta1,beta2);
    if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
        set_dueling_categorical_dqn_training_edge_popup(r->online_net,k_percentage);
        set_dueling_categorical_dqn_training_edge_popup(r->target_net,k_percentage);
    }
    else{
        set_dueling_categorical_dqn_training_gd(r->online_net);
        set_dueling_categorical_dqn_training_gd(r->target_net);
    }
    return r;
}

void free_rainbow(rainbow* r){
    if(r == NULL)
        return;
    uint i;
    free_matrix((void**)r->buffer_state_t,r->max_buffer_size);
    free_matrix((void**)r->buffer_state_t_1,r->max_buffer_size);
    free_matrix((void**)r->diversity_driven_states,r->diversity_driven_q_functions);
    free(r->error_priorization);
    free(r->error_indices);
    free(r->reverse_error_indices);
    free(r->diversity_driven_q_functions_buffer);
    free(r->rewards);
    free(r->ranked_values);
    free(r->recursive_cumulative_ranked_values);
    free(r->nonterminal_state_t_1);
    free(r->actions);
    free(r->last_errors_dqn);
    free(r->last_errors_diversity_driven);
    free(r->positive_rewards);
    free(r->negative_rewards);
    free(r->neutral_rewards);
    free(r->positive_batch);
    free(r->negative_batch);
    free(r->neutral_batch);
    free(r->uniform_sampling_indices);
    // cython could cause problem when dereferencing rainbow object as well as dueling categorical dqns
    //free_dueling_categorical_dqn(r->online_net);
    //free_dueling_categorical_dqn(r->target_net);
    //for(i = 0; i < r->threads; i++){
    //    free_dueling_categorical_dqn_without_learning_parameters(r->online_net_wlp[i]);
    //    free_dueling_categorical_dqn_without_learning_parameters(r->target_net_wlp[i]);
    //}
    //free(r->online_net_wlp);
    //free(r->target_net_wlp);
    
    free(r->batch);
    free(r->reverse_batch);
    free(r->temp_states_t);
    free(r->temp_states_t_1);
    free(r->temp_diversity_states_t);
    free(r->qs);
    free(r->temp_nonterminal_state_t_1);
    free(r->temp_actions);
    free(r->temp_rewards);
    free(r->new_errors);
    free(r->weighted_errors);
    free(r->array_to_shuffle);
    free(r->positive_ranked_values);
    free(r->positive_recursive_cumulative_ranked_values);
    free(r->negative_ranked_values);
    free(r->negative_recursive_cumulative_ranked_values);
    free(r->neutral_ranked_values);
    free(r->neutral_recursive_cumulative_ranked_values);
    free(r->positive_reverse_indices);
    free(r->negative_reverse_indices);
    free(r->neutral_reverse_indices);
    
    
    free(r);
    return;
}

void update_exploration_probability(rainbow* r){
    r->epsilon *= exp(-r->epsilon_decay);
    if(r->epsilon < r->min_epsilon)
        r->epsilon = r->min_epsilon;
    if(r->epsilon > r->max_epsilon)
        r->epsilon = r->max_epsilon;
    return;
}


int get_action_rainbow(rainbow* r, float* state_t, int input_size, int free_state){
    float p = r2();
    if(r->action_taken_iteration < r->stop_epsilon_greedy && p <= r->epsilon){
        r->action_taken_iteration++;
        if(free_state)
            free(state_t);
        int n = rand()%r->online_net->action_size;
        return n;
    }
    else{
        inference_dqn(r->online_net);
        compute_probability_distribution(state_t , input_size, r->online_net);
        int action = argmax(compute_q_functions(r->online_net),r->online_net->action_size);
        train_dqn(r->online_net);
        if(r->action_taken_iteration >= r->stop_epsilon_greedy){
            copy_array(r->online_net->q_functions,r->diversity_driven_q_functions_buffer+r->diversity_driven_q_functions_counter*r->online_net->action_size,r->online_net->action_size);
            free(r->diversity_driven_states[r->diversity_driven_q_functions_counter]);
            r->diversity_driven_states[r->diversity_driven_q_functions_counter] = state_t;
            r->diversity_driven_q_functions_counter++;
            r->diversity_driven_q_functions_counter = r->diversity_driven_q_functions_counter%r->diversity_driven_q_functions;
        }
        
        else if(free_state)
            free(state_t);
            
        
        reset_dueling_categorical_dqn(r->online_net);
        
        return action;
    }
}

void add_state_plus_q_functions_to_diversity_driven_buffer(rainbow* r, float* state_t, float* q_functions){
    copy_array(q_functions,r->diversity_driven_q_functions_buffer+r->diversity_driven_q_functions_counter*r->online_net->action_size,r->online_net->action_size);
    free(r->diversity_driven_states[r->diversity_driven_q_functions_counter]);
    r->diversity_driven_states[r->diversity_driven_q_functions_counter] = state_t;
    r->diversity_driven_q_functions_counter++;
    r->diversity_driven_q_functions_counter = r->diversity_driven_q_functions_counter%r->diversity_driven_q_functions;
}

// for ranked based
// add the error to the buffer given a new buffer and since we didn't fill this part of the buffer
void add_buffer_state(rainbow* r, uint index){
    if(!index)
        r->error_priorization[index] = 100;// get best error
    else
        r->error_priorization[index] = r->error_priorization[r->error_indices[0]];// get best error
    r->error_indices[index] = index;// set the index of this buffer
    r->reverse_error_indices[index] = index;// setting the reverse indexing for error indices
    max_heapify_up(r->error_priorization,r->error_indices,r->reverse_error_indices,index+1,index);//rebuilt the max heap
    
    r->ranked_values[index] = pow(1.0/((double)(index+1)),r->alpha_priorization);
    r->sum_error_priorization_buffer += r->ranked_values[index];// ranked based priorization
    update_recursive_cumulative_heap_up(r->recursive_cumulative_ranked_values, index, 1, index+1, r->ranked_values[index]);
    /*
    int i;
    printf("its error:\n");
    for(i = 0; i < index+1; i++){
        printf("%f ",r->error_priorization[i]);
    }
    printf("\n");
    printf("its rank:\n");
    for(i = 0; i < index+1; i++){
        printf("%d ",r->error_indices[i]);
    }
    printf("\n");
    printf("its probability value:\n");
    for(i = 0; i < index+1; i++){
        printf("%f ",r->ranked_values[i]/r->sum_error_priorization_buffer);
    }
    printf("\nprobability heap array:\n");
    for(i = 0; i < index+1; i++){
        printf("%f ",r->recursive_cumulative_ranked_values[i]/r->sum_error_priorization_buffer);
    }*/
}
void add_buffer_state_reward_sampling(rainbow* r, uint index){
    
    float rew = r->rewards[index];
    if(rew > 0){
        //r->positive_rewards[index] = index;
        r->positive_ranked_values[index] = 1;
        float p = 1;
        update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, index+1, p);
        //r->positive_rewards_counter++;
        r->positive_rewards_length++;
        //r->positive_rewards_counter=r->positive_rewards_counter%r->max_buffer_size;
        r->positive_sum_error_priorization_buffer+=1;
    }
    else if(rew < 0){
        //r->negative_rewards[r->negative_rewards_counter] = index;
        //r->negative_reverse_indices[index] = r->negative_rewards_counter;
        r->negative_ranked_values[index] = 1;
        float p = 1;
        update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, index+1, p);
        //r->negative_rewards_counter++;
        r->negative_rewards_length++;
        //r->negative_rewards_counter=r->negative_rewards_counter%r->max_buffer_size;
        r->negative_sum_error_priorization_buffer+=1;
    }
    else{
        //r->neutral_rewards[r->neutral_rewards_counter] = index;
        //r->neutral_reverse_indices[index] = r->neutral_rewards_counter;
        r->neutral_ranked_values[index] = 1;
        float p = 1;
        update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, index+1, p);
        //r->neutral_rewards_counter++;
        r->neutral_rewards_length++;
        //r->neutral_rewards_counter=r->neutral_rewards_counter%r->max_buffer_size;
        r->neutral_sum_error_priorization_buffer+=1;
    }
}

// for ranked based
void update_buffer_state(rainbow* r, uint index, float error){
    uint length = r->max_buffer_size;
    if(r->buffer_state_t[length-1] == NULL)
        length = r->buffer_current_index;
    remove_ith_element_from_max_heap(r->error_priorization,r->error_indices,r->reverse_error_indices,length,index);
    r->error_priorization[r->error_indices[length-1]] = error;
    max_heapify_up(r->error_priorization,r->error_indices,r->reverse_error_indices,length,length-1);
    /*
    int i;
    printf("error new: %f\n",error);
    printf("its error:\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%f ",r->error_priorization[i]);
    }
    printf("\n");
    printf("its rank:\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%d ",r->error_indices[i]);
    }
    printf("\n");
    printf("its probability value:\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%f ",r->ranked_values[i]/r->sum_error_priorization_buffer);
    }
    printf("\nprobability heap array:\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%f ",r->recursive_cumulative_ranked_values[i]/r->sum_error_priorization_buffer);
    }
    printf("\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%d ",r->error_indices[i]);
    }
    printf("\n");
    for(i = 0; i < r->max_buffer_size; i++){
        printf("%d ",r->reverse_error_indices[i]);
    }
    * */
}

void update_buffer_state_reward_sampling(rainbow* r, uint index, float previous_reward){
    uint length = r->max_buffer_size;
    if(r->buffer_state_t[length-1] == NULL)
        length = r->buffer_current_index;
    float reward = r->rewards[index];
    
    if(previous_reward > 0){
        if(reward > 0){
            float prev = pow(1.0/((double)(r->positive_ranked_values[index])),r->alpha_priorization);
            float p = 1-prev;
            r->positive_sum_error_priorization_buffer+=p;
            r->positive_ranked_values[index] = 1;
            update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, length, p);
        }
        else if(reward < 0){
            float prev = pow(1.0/((double)(r->positive_ranked_values[index])),r->alpha_priorization);
            r->positive_sum_error_priorization_buffer-=prev;
            r->positive_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->positive_rewards_length--;
            //r->negative_rewards[r->negative_rewards_counter] = index;
            //r->negative_reverse_indices[index] = r->negative_rewards_counter;
            r->negative_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, length, p);
            r->negative_rewards_length++;
            r->negative_sum_error_priorization_buffer+=1;
            
        }
        else{
            float prev = pow(1.0/((double)(r->positive_ranked_values[index])),r->alpha_priorization);
            r->positive_sum_error_priorization_buffer-=prev;
            r->positive_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->positive_rewards_length--;
            //r->neutral_rewards[r->neutral_rewards_counter] = index;
            //r->neutral_reverse_indices[index] = r->neutral_rewards_counter;
            r->neutral_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, length, p);
            r->neutral_rewards_length++;
            //r->negative_rewards_counter=r->negative_rewards_counter%r->max_buffer_size;
            r->neutral_sum_error_priorization_buffer+=1;
            
        }
    }
    else if(previous_reward < 0){
        if(reward < 0){
            float prev = pow(1.0/((double)(r->negative_ranked_values[index])),r->alpha_priorization);
            float p = 1-prev;
            r->negative_sum_error_priorization_buffer+=p;
            r->negative_ranked_values[index] = 1;
            update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, length, p);
        }
        else if(reward > 0){
            float prev = pow(1.0/((double)(r->negative_ranked_values[index])),r->alpha_priorization);
            r->negative_sum_error_priorization_buffer-=prev;
            r->negative_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->negative_rewards_length--;
            //r->negative_rewards[r->negative_rewards_counter] = index;
            //r->negative_reverse_indices[index] = r->negative_rewards_counter;
            r->positive_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, length, p);
            r->positive_rewards_length++;
            r->positive_sum_error_priorization_buffer+=1;
            
        }
        else{
            float prev = pow(1.0/((double)(r->negative_ranked_values[index])),r->alpha_priorization);
            r->negative_sum_error_priorization_buffer-=prev;
            r->negative_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->negative_rewards_length--;
            //r->neutral_rewards[r->neutral_rewards_counter] = index;
            //r->neutral_reverse_indices[index] = r->neutral_rewards_counter;
            r->neutral_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, length, p);
            r->neutral_rewards_length++;
            //r->negative_rewards_counter=r->negative_rewards_counter%r->max_buffer_size;
            r->neutral_sum_error_priorization_buffer+=1;
            
        }
    }
    else{
        if(reward == 0){
            float prev = pow(1.0/((double)(r->neutral_ranked_values[index])),r->alpha_priorization);
            float p = 1-prev;
            r->neutral_sum_error_priorization_buffer+=p;
            r->neutral_ranked_values[index] = 1;
            update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, length, p);
        }
        else if(reward > 0){
            float prev = pow(1.0/((double)(r->neutral_ranked_values[index])),r->alpha_priorization);
            r->neutral_sum_error_priorization_buffer-=prev;
            r->neutral_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->neutral_rewards_length--;
            //r->negative_rewards[r->negative_rewards_counter] = index;
            //r->negative_reverse_indices[index] = r->negative_rewards_counter;
            r->positive_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, index, 1, length, p);
            r->positive_rewards_length++;
            r->positive_sum_error_priorization_buffer+=1;
            
        }
        else{
            float prev = pow(1.0/((double)(r->neutral_ranked_values[index])),r->alpha_priorization);
            r->neutral_sum_error_priorization_buffer-=prev;
            r->neutral_ranked_values[index] = 0;
            update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, index, 1, length, -prev);
            r->neutral_rewards_length--;
            //r->negative_rewards[r->negative_rewards_counter] = index;
            //r->negative_reverse_indices[index] = r->negative_rewards_counter;
            r->negative_ranked_values[index] = 1;
            float p = 1;
            update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, index, 1, length, p);
            r->negative_rewards_length++;
            r->negative_sum_error_priorization_buffer+=1;
            
        }
    }
}



void add_experience(rainbow* r, float* state_t, float* state_t_1, int action, float reward, int nonterminal_s_t_1){
    int was_null = 1;
    float previous_reward = 0;
    if(r->buffer_state_t[r->buffer_current_index] != NULL){
        was_null = 0;
        previous_reward = r->rewards[r->buffer_current_index];
    }
    free(r->buffer_state_t[r->buffer_current_index]);
    free(r->buffer_state_t_1[r->buffer_current_index]);
    r->buffer_state_t[r->buffer_current_index] = state_t;
    r->buffer_state_t_1[r->buffer_current_index] = state_t_1;
    r->rewards[r->buffer_current_index] = reward;
    r->actions[r->buffer_current_index] = action;
    r->nonterminal_state_t_1[r->buffer_current_index] = nonterminal_s_t_1;
    
    int i;
    /*
    printf("c code\n");
    for(i = 0; i < get_input_layer_size_dueling_categorical_dqn(r->online_net); i++){
        printf("%f ",r->buffer_state_t[r->buffer_current_index][i]);
    }
    printf("\n");
    for(i = 0; i < get_input_layer_size_dueling_categorical_dqn(r->online_net); i++){
        printf("%f ",r->buffer_state_t_1[r->buffer_current_index][i]);
    }
    printf("\n");
    printf("%d\n",action);
    printf("%f\n",reward);
    printf("%d\n",nonterminal_s_t_1);
    */
    
    if(r->sampling_flag == RANKED_SAMPLING){
        if(was_null)
            add_buffer_state(r, r->buffer_current_index);
        
        else
            update_buffer_state(r, r->reverse_error_indices[r->buffer_current_index], r->error_priorization[r->error_indices[0]]);
    }
    
    else if(r->sampling_flag == REWARD_SAMPLING){
        //printf("counter: %d\n",r->buffer_current_index);
        if(was_null)
            add_buffer_state_reward_sampling(r, r->buffer_current_index);
        
        else
            update_buffer_state_reward_sampling(r, r->buffer_current_index, previous_reward);
    }
    
    else if(was_null && r->sampling_flag == UNIFORM_SAMPLING){
        r->uniform_sampling_indices[r->buffer_current_index] = r->buffer_current_index;
    }
    
  
    r->buffer_current_index++;
    r->buffer_current_index = r->buffer_current_index%r->max_buffer_size;
}

void train_rainbow(rainbow* r, int last_t_1_was_terminal){
    // only after each episode we train
    if(!last_t_1_was_terminal)
        return;
    // checking the buffer current size    
    uint length = r->max_buffer_size;
    if(r->buffer_state_t[length-1] == NULL)
        length = r->buffer_current_index;
    // if the buffer current size is less then the batch size we don't train yet
    if(length < r->batch_size)
        return;
    // prioritized sampling (ranked based for simplicity)
    uint i,j,index;
    double over_sum = r->sum_error_priorization_buffer;
    if(r->sampling_flag == RANKED_SAMPLING){
        for(i = 0; i < r->batch_size; i++){
            float p = r2();
            uint val = weighted_random_sample(r->recursive_cumulative_ranked_values,r->ranked_values,0,length,p,over_sum, r->reverse_batch, i);
            r->batch[i] = r->error_indices[val];
            r->reverse_batch[i] = val;
            if(!index_is_inside_buffer(r->reverse_batch,i,val)){
                over_sum-=r->ranked_values[val];
            }
        }
    }
    if(r->sampling_flag == REWARD_SAMPLING){
        int positive_rewards =0;
        int negative_rewards = 0;
        int neutral_rewards =0;
        double negative_sum = r->negative_sum_error_priorization_buffer;
        double positive_sum = r->positive_sum_error_priorization_buffer;
        double neutral_sum = r->neutral_sum_error_priorization_buffer;
        
        for(i = 0; i < r->batch_size; i++){
            float samp = r2();
            float p = r2();
            double prob_sum = 0;
            double neg_prob_sum = 0, pos_prob_sum = 0, neutr_prob_sum = 0;
            if(negative_rewards < r->negative_rewards_length){
                neg_prob_sum = (1.0/((double)r->negative_rewards_length));
            }
            if(positive_rewards < r->positive_rewards_length){
                pos_prob_sum = (1.0/((double)r->positive_rewards_length));
            }
            if(neutral_rewards < r->neutral_rewards_length){
                neutr_prob_sum = (1.0/((double)r->neutral_rewards_length));
            }
            
            neg_prob_sum = pow(neg_prob_sum,r->beta_priorization);
            pos_prob_sum = pow(pos_prob_sum,r->beta_priorization);
            neutr_prob_sum = pow(neutr_prob_sum,r->beta_priorization);
            prob_sum = neg_prob_sum+pos_prob_sum+neutr_prob_sum;
            neg_prob_sum = neg_prob_sum/prob_sum;
            pos_prob_sum = pos_prob_sum/prob_sum;
            neutr_prob_sum = neutr_prob_sum/prob_sum;

            if(negative_rewards < r->negative_rewards_length){
                if(samp <= neg_prob_sum){
                    uint val = weighted_random_sample_rewards(r->negative_recursive_cumulative_ranked_values,r->negative_ranked_values,0,r->max_buffer_size,p,negative_sum,r->negative_batch,negative_rewards,r->alpha_priorization);
                    if (!index_is_inside_buffer(r->batch,i,val))
                        negative_sum-=pow(((double)1.0/((double)r->negative_ranked_values[val])),r->alpha_priorization);
                    r->batch[i] = val;
                    r->negative_batch[negative_rewards] = val;
                    negative_rewards++;
                }
                else if(positive_rewards < r->positive_rewards_length){
                    if(samp-neg_prob_sum <= pos_prob_sum){
                        uint val = weighted_random_sample_rewards(r->positive_recursive_cumulative_ranked_values,r->positive_ranked_values,0,r->max_buffer_size,p,positive_sum,r->positive_batch,positive_rewards,r->alpha_priorization);
                        if (!index_is_inside_buffer(r->batch,i,val))
                            positive_sum-=pow(((double)1.0/((double)r->positive_ranked_values[val])),r->alpha_priorization);
                        r->batch[i] = val;
                        r->positive_batch[positive_rewards] = val;
                        positive_rewards++;
                    }
                    else{
                        uint val = weighted_random_sample_rewards(r->neutral_recursive_cumulative_ranked_values,r->neutral_ranked_values,0,r->max_buffer_size,p,neutral_sum,r->neutral_batch,neutral_rewards,r->alpha_priorization);
                        if (!index_is_inside_buffer(r->batch,i,val))
                            neutral_sum-=pow(((double)1.0/((double)r->neutral_ranked_values[val])),r->alpha_priorization);
                        r->batch[i] = val;
                        r->neutral_batch[neutral_rewards] = val;
                        neutral_rewards++;
                    }
                }
                else{
                    uint val = weighted_random_sample_rewards(r->neutral_recursive_cumulative_ranked_values,r->neutral_ranked_values,0,r->max_buffer_size,p,neutral_sum,r->neutral_batch,neutral_rewards,r->alpha_priorization);
                    if (!index_is_inside_buffer(r->batch,i,val))
                        neutral_sum-=pow(((double)1.0/((double)r->neutral_ranked_values[val])),r->alpha_priorization);
                    r->batch[i] = val;
                    r->neutral_batch[neutral_rewards] = val;
                    neutral_rewards++;
                }    
            }
            else if(positive_rewards < r->positive_rewards_length){
                if(samp <= pos_prob_sum){
                    uint val = weighted_random_sample_rewards(r->positive_recursive_cumulative_ranked_values,r->positive_ranked_values,0,r->max_buffer_size,p,positive_sum,r->positive_batch,positive_rewards,r->alpha_priorization);
                    if (!index_is_inside_buffer(r->batch,i,val))
                        positive_sum-=pow(((double)1.0/((double)r->positive_ranked_values[val])),r->alpha_priorization);
                    r->batch[i] = val;
                    r->positive_batch[positive_rewards] = val;
                    positive_rewards++;
                }
                else{
                    uint val = weighted_random_sample_rewards(r->neutral_recursive_cumulative_ranked_values,r->neutral_ranked_values,0,r->max_buffer_size,p,neutral_sum,r->neutral_batch,neutral_rewards,r->alpha_priorization);
                    if (!index_is_inside_buffer(r->batch,i,val))    
                        neutral_sum-=pow(((double)1.0/((double)r->neutral_ranked_values[val])),r->alpha_priorization);
                    r->batch[i] = val;
                    r->neutral_batch[neutral_rewards] = val;
                    neutral_rewards++;
                }
            }
            else{
                uint val = weighted_random_sample_rewards(r->neutral_recursive_cumulative_ranked_values,r->neutral_ranked_values,0,r->max_buffer_size,p,neutral_sum,r->neutral_batch,neutral_rewards,r->alpha_priorization);
                if (!index_is_inside_buffer(r->batch,i,val))
                    neutral_sum-=pow(((double)1.0/((double)r->neutral_ranked_values[val])),r->alpha_priorization);
                r->batch[i] = val;
                r->neutral_batch[neutral_rewards] = val;
                neutral_rewards++;
            }
        }
    }
    else if(r->sampling_flag == UNIFORM_SAMPLING){
        shuffle_int_array_until_length(r->uniform_sampling_indices,length,r->batch_size);
        for(i = 0; i < r->batch_size; i++){
            r->batch[i] = r->uniform_sampling_indices[i];
        }
    }
    // n step forward sampling
    for(i = 0; i < r->batch_size; i++){
        r->temp_states_t_1[i] = NULL;
        //printf("batch: %d\n",r->batch[i]);
        r->temp_states_t[i] = r->buffer_state_t[r->batch[i]];
        float reward = 0;
        float lambda = 1;
        for(j = 0; j < r->n_step_rewards; j++){
            index = (r->batch[i]+j)%r->max_buffer_size;
            reward+=lambda*r->rewards[index];
            if(!r->nonterminal_state_t_1[index]){
                //printf("uzzioo\n");
                r->temp_states_t_1[i] = r->buffer_state_t_1[index];
                r->temp_nonterminal_state_t_1[i] = r->nonterminal_state_t_1[index];
                break;
            }
            //printf("lezz\n");
            lambda*=r->lambda_value;
        }
        if(r->temp_states_t_1[i] == NULL){
            j--;
            index = (r->batch[i]+j)%r->max_buffer_size;
            r->temp_states_t_1[i] = r->buffer_state_t_1[index];
            r->temp_nonterminal_state_t_1[i] = r->nonterminal_state_t_1[index];
        }
        r->temp_actions[i] = r->actions[r->batch[i]];
        r->temp_rewards[i] = reward;
        if(r->sampling_flag == RANKED_SAMPLING)
            r->weighted_errors[i] = pow(1.0/(((float)(length))*(r->ranked_values[r->batch[i]]/r->sum_error_priorization_buffer)),r->beta_priorization);
        else
            r->weighted_errors[i] = 1;
    }
    if(r->sampling_flag == RANKED_SAMPLING){
        // maximum weight
        float maximum = 1.0/(pow(1.0/(((float)(length))*(r->ranked_values[length-1]/r->sum_error_priorization_buffer)),r->beta_priorization));
        mul_value(r->weighted_errors,maximum,r->weighted_errors,r->batch_size);
    }
    
    // first td error 
    for(i = 0; i < r->batch_size; i+=r->threads){
        int min = r->threads;
        if(r->batch_size-i < min)
            min = r->batch_size-i;
        dueling_categorical_dqn_train_with_error(min,r->online_net,r->target_net,r->online_net_wlp,r->target_net_wlp,r->temp_states_t+i,r->temp_rewards+i,r->temp_actions+i,r->temp_states_t_1+i,r->temp_nonterminal_state_t_1+i,r->gamma,get_input_layer_size_dueling_categorical_dqn(r->online_net),r->new_errors+i,r->weighted_errors+i);
        sum_dueling_categorical_dqn_partial_derivatives_multithread(r->online_net_wlp,r->online_net,min,0);// log n
        dueling_categorical_reset_without_learning_parameters_reset(r->online_net_wlp,min);
        dueling_categorical_reset_without_learning_parameters_reset(r->target_net_wlp,min);
    }
    
    // now dd error
    // uniform random sampling (we could maybe sample with ranked based priorization as the td sampling is done, maybe future implementation)
    float real_ret = 0;
    if(length >= r->diversity_driven_q_functions){
        float ret = 0;
        int flag = 1;
        if(r->diversity_driven_states[r->diversity_driven_q_functions-1] == NULL){
            if(r->diversity_driven_q_functions_counter < r->batch_size)
                flag = 0;
            else{
                for(i = 0; i < r->diversity_driven_q_functions; i++){
                    r->array_to_shuffle[i] = i;
                }
                shuffle_int_array(r->array_to_shuffle,r->diversity_driven_q_functions_counter);
            }
            
        }
        else
            shuffle_int_array(r->array_to_shuffle,r->diversity_driven_q_functions);
        
        if(flag){
            
            for(i = 0; i < r->batch_size; i++){
                r->temp_diversity_states_t[i] = r->diversity_driven_states[r->array_to_shuffle[i]];
                r->qs[i] = &r->diversity_driven_q_functions_buffer[r->array_to_shuffle[i]];
            }
            for(i = 0; i < r->batch_size; i+=r->threads){
                int min = r->threads;
                if(r->batch_size-i < min)
                    min = r->batch_size-i;
                ret+=dueling_categorical_dqn_train_kl(min,r->online_net,r->online_net_wlp,r->temp_diversity_states_t+i,r->qs+i,1,r->alpha,r->clipping_gradient_value);
                sum_dueling_categorical_dqn_partial_derivatives_multithread(r->online_net_wlp,r->online_net,min,0);// log n
                dueling_categorical_reset_without_learning_parameters_reset(r->online_net_wlp,min);
            }
            ret/=r->batch_size;
            if(ret < 0)
                ret = -ret;
            if(ret < r->diversity_driven_threshold)
                r->alpha*=1.01;
            else
                r->alpha*=0.99;
            }
            r->diversity_driven_threshold*=exp(-r->diversity_driven_decay);
            if(r->diversity_driven_threshold < r->diversity_driven_minimum)
                r->diversity_driven_threshold = r->diversity_driven_minimum;
            if(r->diversity_driven_threshold > r->diversity_driven_maximum)
                r->diversity_driven_threshold = r->diversity_driven_maximum;
    }
    // adaptive clipping gradient
    if(r->adaptive_clipping_flag)
        adaptive_gradient_clipping_dueling_categorical_dqn(r->online_net,r->adaptive_clipping_gradient_value,1e-3);
    // update and reset online net
    update_dueling_categorical_dqn(r->online_net,r->lr,r->momentum,r->batch_size,r->gd_flag,&r->beta1,&r->beta2,NO_REGULARIZATION,0,0,(unsigned long long int*)&r->train_iteration);
    
    // copy for dobule dqn
    if(r->train_iteration && !(r->train_iteration%r->epochs_to_copy_target)){
        slow_paste_dueling_categorical_dqn(r->online_net,r->target_net,r->tau_copying);
    }
    // update learning rate
    update_lr(&r->lr,r->lr_minimum,r->lr_maximum,r->initial_lr,r->lr_decay,r->train_iteration,r->lr_epoch_threshold,r->lr_decay_flag);
    // update training parameters
    update_training_parameters(&r->beta1,&r->beta2,(unsigned long long int*)&r->train_iteration,r->online_net->shared_hidden_layers->beta1_adam,r->online_net->shared_hidden_layers->beta2_adam);
    reset_dueling_categorical_dqn(r->online_net);
    reset_dueling_categorical_dqn(r->target_net);
    // if ranked based sampling we must update the errors
    if(r->sampling_flag == RANKED_SAMPLING){
        for(i = 0; i < r->batch_size; i++){
            if(!index_is_inside_buffer(r->reverse_batch,i,r->reverse_batch[i])){
                float err = r->new_errors[i];
                if(err <0)
                    err = -err;
                update_buffer_state(r,r->reverse_batch[i],err);
            }
        }
        r->beta_priorization*=exp(r->beta_priorization_increase);
        if(r->beta_priorization > 1)
            r->beta_priorization = 1;
    }
    else if(r->sampling_flag == REWARD_SAMPLING){
        for(i = 0; i < r->batch_size; i++){
            if(!index_is_inside_buffer(r->batch,i,r->batch[i])){
                if(r->rewards[r->batch[i]] > 0){
                    float old = pow(((double)1.0/((double)r->positive_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->positive_ranked_values[r->batch[i]]++;
                    float new = pow(((double)1.0/((double)r->positive_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->positive_sum_error_priorization_buffer-=old;
                    r->positive_sum_error_priorization_buffer+=new;
                    float p = new-old;
                    update_recursive_cumulative_heap_up(r->positive_recursive_cumulative_ranked_values, r->batch[i], 1, length, p);
                    
                }
                else if(r->rewards[r->batch[i]] < 0){
                    float old = pow(((double)1.0/((double)r->negative_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->negative_ranked_values[r->batch[i]]++;
                    float new = pow(((double)1.0/((double)r->negative_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->negative_sum_error_priorization_buffer-=old;
                    r->negative_sum_error_priorization_buffer+=new;
                    float p = new-old;
                    update_recursive_cumulative_heap_up(r->negative_recursive_cumulative_ranked_values, r->batch[i], 1, length, p);
                    
                }
                else{
                    float old = pow(((double)1.0/((double)r->neutral_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->neutral_ranked_values[r->batch[i]]++;
                    float new = pow(((double)1.0/((double)r->neutral_ranked_values[r->batch[i]])),r->alpha_priorization);
                    r->neutral_sum_error_priorization_buffer-=old;
                    r->neutral_sum_error_priorization_buffer+=new;
                    float p = new-old;
                    update_recursive_cumulative_heap_up(r->neutral_recursive_cumulative_ranked_values, r->batch[i], 1, length, p);
                }
            }
        }
    }
    
}


