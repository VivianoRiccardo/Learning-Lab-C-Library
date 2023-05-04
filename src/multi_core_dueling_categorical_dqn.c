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


void* dueling_categorical_dqn_train_thread(void* _args) {
    // depacking args
    thread_args_dueling_categorical_dqn_train* dqn = (thread_args_dueling_categorical_dqn_train*) _args;
    float* error;
    if(!dqn->online_net->is_qr){
        error = get_loss_for_dueling_categorical_dqn_opt(dqn->online_net,dqn->online_net_wlp,dqn->target_net,dqn->target_net_wlp,dqn->state_t,dqn->action_t,dqn->reward_t,dqn->state_t_1,dqn->lambda,dqn->state_sizes,dqn->nonterminal_s_t_1);
        bp_dueling_categorical_network_opt(dqn->state_t,dqn->state_sizes,error,dqn->online_net,dqn->online_net_wlp);
    }
    else{
        get_loss_for_qr_dqn_opt(dqn->online_net,dqn->online_net_wlp,dqn->target_net,dqn->target_net_wlp,dqn->state_t,dqn->action_t,dqn->reward_t,dqn->state_t_1,dqn->lambda,dqn->state_sizes,dqn->nonterminal_s_t_1);
        bp_qr_dqn_opt(dqn->state_t,dqn->state_sizes,error,dqn->online_net,dqn->online_net_wlp);
    }
    return _args;
}

void* dueling_categorical_dqn_train_kl_thread(void* _args) {
    // depacking args
    thread_args_dueling_categorical_dqn_train* dqn = (thread_args_dueling_categorical_dqn_train*) _args;
    dqn->ret[0] = compute_kl_dueling_categorical_dqn_opt(dqn->online_net,dqn->online_net_wlp,dqn->state_t,dqn->q_functions,dqn->weight,dqn->alpha,dqn->clip);
    bp_dueling_categorical_network_opt(dqn->state_t,get_input_layer_size_dueling_categorical_dqn(dqn->online_net),dqn->online_net_wlp->error,dqn->online_net,dqn->online_net_wlp);
    return _args;
}

void* dueling_categorical_dqn_train_l1_thread(void* _args) {
    // depacking args
    thread_args_dueling_categorical_dqn_train* dqn = (thread_args_dueling_categorical_dqn_train*) _args;
    if(!dqn->online_net->is_qr){
        dqn->ret[0] = compute_l1_dueling_categorical_dqn_opt(dqn->online_net,dqn->online_net_wlp,dqn->state_t,dqn->q_functions,dqn->weight,dqn->alpha,dqn->clip);
        bp_dueling_categorical_network_opt(dqn->state_t,get_input_layer_size_dueling_categorical_dqn(dqn->online_net),dqn->online_net_wlp->error,dqn->online_net,dqn->online_net_wlp);
    }
    else{
        dqn->ret[0] = compute_l1_qr_dqn_opt(dqn->online_net,dqn->online_net_wlp,dqn->state_t,dqn->q_functions,dqn->weight,dqn->alpha,dqn->clip);
        bp_qr_dqn_opt(dqn->state_t,get_input_layer_size_dueling_categorical_dqn(dqn->online_net),dqn->online_net_wlp->error,dqn->online_net,dqn->online_net_wlp);
    }
    return _args;
}


void* dueling_categorical_dqn_train_with_error_thread(void* _args) {
    // depacking args
    thread_args_dueling_categorical_dqn_train* dqn = (thread_args_dueling_categorical_dqn_train*) _args;
    float* error ;
    if(!dqn->online_net->is_qr){
        error = get_loss_for_dueling_categorical_dqn_opt_with_error(dqn->online_net,dqn->online_net_wlp,dqn->target_net,dqn->target_net_wlp,dqn->state_t,dqn->action_t,dqn->reward_t,dqn->state_t_1,dqn->lambda,dqn->state_sizes,dqn->nonterminal_s_t_1,dqn->new_error,dqn->weighted_error);
        bp_dueling_categorical_network_opt(dqn->state_t,dqn->state_sizes,error,dqn->online_net,dqn->online_net_wlp);
    }
    else{
        //printf("multi thread qrdqn\n");
        error = get_loss_for_qr_dqn_opt_with_error(dqn->online_net,dqn->online_net_wlp,dqn->target_net,dqn->target_net_wlp,dqn->state_t,dqn->action_t,dqn->reward_t,dqn->state_t_1,dqn->lambda,dqn->state_sizes,dqn->nonterminal_s_t_1,dqn->new_error,dqn->weighted_error);
        bp_qr_dqn_opt(dqn->state_t,dqn->state_sizes,error,dqn->online_net,dqn->online_net_wlp);
    }
    return _args;
}

void* dueling_categorical_dqn_reset_thread(void* _args) {
    // depacking args
    thread_args_dueling_categorical_dqn_train* dqn = (thread_args_dueling_categorical_dqn_train*) _args;
    reset_dueling_categorical_dqn_without_learning_parameters(dqn->online_net);
    return _args;
}

void* dueling_categorical_dqn_thread_sum(void* _args) {
    
    // depacking args
    thread_args_dueling_categorical_dqn* args = (thread_args_dueling_categorical_dqn*) _args;
    sum_dueling_categorical_dqn_partial_derivatives_multithread(args->m, args->m[0], args->n, args->depth);
    return _args;
}


void dueling_categorical_dqn_train(int threads, dueling_categorical_dqn* online_net,dueling_categorical_dqn* target_net, dueling_categorical_dqn** online_net_wlp, dueling_categorical_dqn** target_net_wlp, float** states_t, float* rewards_t, int* actions_t, float** states_t_1, int* nonterminals_t_1, float lambda_value, int state_sizes){
    pthread_t thread[threads];
    thread_args_dueling_categorical_dqn_train* args[threads];
    
    int j;
    
    for(j = 0; j < threads; j++){
        args[j] = (thread_args_dueling_categorical_dqn_train*)malloc(sizeof(thread_args_dueling_categorical_dqn_train));
        args[j]->online_net = online_net;
        args[j]->online_net_wlp = online_net_wlp[j];
        args[j]->target_net = target_net;
        args[j]->target_net_wlp = target_net_wlp[j];
        args[j]->state_t = states_t[j];
        args[j]->reward_t = rewards_t[j];
        args[j]->action_t = actions_t[j];
        args[j]->nonterminal_s_t_1 = nonterminals_t_1[j];
        args[j]->state_t_1 = states_t_1[j];
        args[j]->lambda = lambda_value;
        args[j]->state_sizes = state_sizes;
        pthread_create(thread+j, NULL, dueling_categorical_dqn_train_thread, args[j]);
    }
        
                
    for(j = 0; j < threads; j++) {
        pthread_join(thread[j], NULL);
        free(args[j]);
    }
    return;

}

float dueling_categorical_dqn_train_kl(int threads, dueling_categorical_dqn* online_net, dueling_categorical_dqn** online_net_wlp, float** states_t, float** q_functions, float weight, float alpha, float clip){
    pthread_t thread[threads];
    thread_args_dueling_categorical_dqn_train* args[threads];
    
    int j;
    float ret = 0;
    for(j = 0; j < threads; j++){
        float error = 0;
        args[j] = (thread_args_dueling_categorical_dqn_train*)malloc(sizeof(thread_args_dueling_categorical_dqn_train));
        args[j]->online_net = online_net;
        args[j]->online_net_wlp = online_net_wlp[j];
        args[j]->state_t = states_t[j];
        args[j]->q_functions = q_functions[j];
        args[j]->weight = weight;
        args[j]->clip = clip;
        args[j]->alpha = alpha;
        args[j]->ret = &error;
        pthread_create(thread+j, NULL, dueling_categorical_dqn_train_kl_thread, args[j]);
    }
        
                
    for(j = 0; j < threads; j++) {
        pthread_join(thread[j], NULL);
        ret+=args[j]->ret[0];
        free(args[j]);
    }
    return ret;

}


float dueling_categorical_dqn_train_l1(int batch_size, int threads, dueling_categorical_dqn* online_net, dueling_categorical_dqn** online_net_wlp, float** states_t, float** q_functions, float weight, float alpha, float clip){
    pthread_t thread[threads];
    thread_args_dueling_categorical_dqn_train* args[threads];
    
    int i,j;
    float ret = 0;
    for(i = 0; i < batch_size; i+=threads){
        int min = threads;
        if(batch_size-i < threads)
            min = batch_size-i;
        for(j = 0; j < min; j++){
            float error = 0;
            args[j] = (thread_args_dueling_categorical_dqn_train*)malloc(sizeof(thread_args_dueling_categorical_dqn_train));
            args[j]->online_net = online_net;
            args[j]->online_net_wlp = online_net_wlp[j];
            args[j]->state_t = states_t[j];
            args[j]->q_functions = q_functions[j];
            args[j]->weight = weight;
            args[j]->clip = clip;
            args[j]->alpha = alpha;
            args[j]->ret = &error;
            pthread_create(thread+j, NULL, dueling_categorical_dqn_train_l1_thread, args[j]);
        }
            
                    
        for(j = 0; j < min; j++) {
            pthread_join(thread[j], NULL);
            ret+=args[j]->ret[0];
            free(args[j]);
        }
        sum_dueling_categorical_dqn_partial_derivatives_multithread(online_net_wlp,online_net,min,0);
        dueling_categorical_reset_without_learning_parameters_reset(online_net_wlp,min);
    }
    return ret;

}

void dueling_categorical_dqn_train_with_error(int threads, dueling_categorical_dqn* online_net,dueling_categorical_dqn* target_net, dueling_categorical_dqn** online_net_wlp, dueling_categorical_dqn** target_net_wlp, float** states_t, float* rewards_t, int* actions_t, float** states_t_1, int* nonterminals_t_1, float* lambda_value, int state_sizes, float* new_errors, float* weighted_errors){
    pthread_t thread[threads];
    thread_args_dueling_categorical_dqn_train* args[threads];
    
    int j;
    
    for(j = 0; j < threads; j++){
        args[j] = (thread_args_dueling_categorical_dqn_train*)malloc(sizeof(thread_args_dueling_categorical_dqn_train));
        args[j]->online_net = online_net;
        args[j]->online_net_wlp = online_net_wlp[j];
        args[j]->target_net = target_net;
        args[j]->target_net_wlp = target_net_wlp[j];
        args[j]->state_t = states_t[j];
        args[j]->reward_t = rewards_t[j];
        args[j]->new_error = &new_errors[j];
        args[j]->weighted_error = weighted_errors[j];
        args[j]->action_t = actions_t[j];
        args[j]->nonterminal_s_t_1 = nonterminals_t_1[j];
        args[j]->state_t_1 = states_t_1[j];
        args[j]->lambda = lambda_value[j];
        args[j]->state_sizes = state_sizes;
        pthread_create(thread+j, NULL, dueling_categorical_dqn_train_with_error_thread, args[j]);
    }
        
                
    for(j = 0; j < threads; j++) {
        pthread_join(thread[j], NULL);
        free(args[j]);
    }
    return;

}

dueling_categorical_dqn* sum_dueling_categorical_dqn_partial_derivatives_multithread(dueling_categorical_dqn** batch_m, dueling_categorical_dqn* m, int n, int depth){
    if(depth == 0 && n == 1){
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[0],m);
        return m;
    }
    else if(depth == 0 && n == 2){
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[0],m);
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[1],m);
        return m;
    }
    else if(depth == 0 && n == 3){
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[0],m);
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[1],m);
        sum_dueling_categorical_dqn_partial_derivatives(m,batch_m[2],m);
        return m;
    }
    
    if(n == 0)
        return NULL;
    else if(n == 1)
        return batch_m[0];
    else if(n == 2){
        sum_dueling_categorical_dqn_partial_derivatives(batch_m[0],batch_m[1],batch_m[0]);
        return batch_m[0];
    }
    else if(n == 3){
        sum_dueling_categorical_dqn_partial_derivatives(batch_m[0],batch_m[1],batch_m[0]);
        sum_dueling_categorical_dqn_partial_derivatives(batch_m[0],batch_m[2],batch_m[0]);
        return batch_m[0];
    }
    pthread_t thread[2];
    thread_args_dueling_categorical_dqn* args[2];
    int size = (int)(n/2);
    args[0] = (thread_args_dueling_categorical_dqn*)malloc(sizeof(thread_args_dueling_categorical_dqn));
    args[0]->depth = depth+1;
    args[0]->n = size;
    args[0]->m = batch_m;
    args[1] = (thread_args_dueling_categorical_dqn*)malloc(sizeof(thread_args_dueling_categorical_dqn));
    args[1]->depth = depth+1;
    args[1]->n = n-size;
    args[1]->m = batch_m+size;
    pthread_create(thread, NULL, dueling_categorical_dqn_thread_sum, args[0]);
    pthread_create(thread+1, NULL, dueling_categorical_dqn_thread_sum, args[1]);
    pthread_join(thread[0],NULL);
    pthread_join(thread[1],NULL);
    free(args[0]);
    free(args[1]);
    dueling_categorical_dqn* t1 = batch_m[0];
    dueling_categorical_dqn* t2 = batch_m[size];
    sum_dueling_categorical_dqn_partial_derivatives(t1,t2,t1);
    if(depth == 0){
        sum_dueling_categorical_dqn_partial_derivatives(t1,m,m);
        t1 = m;
    }
    
    return t1;
}

void dueling_categorical_reset_without_learning_parameters_reset(dueling_categorical_dqn** dqn, int threads){
    pthread_t thread[threads];
    thread_args_dueling_categorical_dqn_train* args[threads];
    
    int j;
    
    for(j = 0; j < threads; j++){
        args[j] = (thread_args_dueling_categorical_dqn_train*)malloc(sizeof(thread_args_dueling_categorical_dqn_train));
        args[j]->online_net = dqn[j];
        pthread_create(thread+j, NULL, dueling_categorical_dqn_reset_thread, args[j]);
    }
        
                
    for(j = 0; j < threads; j++) {
        pthread_join(thread[j], NULL);
        free(args[j]);
    }
    return;
}

