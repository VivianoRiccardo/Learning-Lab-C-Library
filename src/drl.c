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

/* This function initializes a ddpg model with 4 models:
 * model m1 is the policy network, model m2 is the network that is gonna handle the frames from the game, m3 is the model that is gonna handle the actions, model m4 is the final part
 * of the q-function model. so m2,m3,m4 are the q function total model
 * regularization1 can be NO_REGULARIZATION or L2_REGULARIZATION for the policy network, regularization2 is for q-function network. gradient descent flag1 can be NESTEROV, ADAM, RADAM and others and is for policy network
 * gradient descent flag2 is for q-function network. buff size is the size of the frames passed, it should be = to m1_input. maxframes is the maximum nuber of frames that the ddpg should handle.
 * lr1 is the learning rate of the policy network, lr2 of the q-function network. momentum1 is the momentum for policy network, momentum2 is for the q-function network. 
 * lambda1 is the value of l2 regularization if there is any l2 regularization of policy network, lambda2 of q-function network. tau is the param used by the target networks to copy the original networks
 * lambda is the param used for the q function algorithm.
 * 
 * Input:
 * 
 *             
 *                 @ model * m1:= the policy network (avoid softmax we are talking of ddpg! no discrete action space man)
 *                 @ model* m2:= critic network. is the part of the critic network that handles the frames of the game
 *                 @ model* m3:= critic network:= is the part of the critic network that handles the actions taken
 *                 @ model* m4:= critic network:= is the final part of critic network that takes as input m2 and m3 output
 *                 @ int batch_size:0 the size of the batch
 *                 @ int threads:= the number of threads you want to use
 *                 @ int regularization1:= the regularization of actor network
 *                 @ int regularization2:= the regularization of critic network
 *                 @ m1_input:= the size of the input for m1
 *                 @ m1_output:= the size of the output of m1
 *                 @ m2_output:= the size of the output of m2
 *                 @ m3_output:= the size of the output of m3
 *                 @ int gradient_descent_flag1:= the optimization algorithm used for actor network
 *                 @ int gradient_descent_flag2:= the optimization algorithm used for critic network
 *                 @ int buff_size:= is equal to m1_input (idk why i added this param, i don't want to change it sorry :p, bad developer here)
 *                 @ int max_frames:= the size of the actions, terminal, rewards... you have 2 options: 1) when you train your ddpg model you can set immediatly
 *                                    max frames equal to your maximum number of frames ex: 5 milions, and handle the terminal actions and reward state from ddpg structure
 *                                    or you can create your own buffer for states, actions, rewards, termnal and set max frames = batch_size and then when you train
 *                                    your ddpg model you copy your frames, actions, rewards, etc in these structures, cause ddpg_train function uses the buffer of this structure
 *                 @ float lr1:= the learning rate of actor network
 *                 @ float lr2:= the learning rate of critic network
 *                 @ float momentum1:= the momentum of actor network
 *                 @ float momentum2:= the momentum of critic network
 *                 @ float tau:= the parameter used by the target networks for both target actor and target critic
 *                 @ float epsilon_greedy:= is useless
 *                 @ float lambda:= is the lambda param used by the q-function (for the god sake go to learn something!)
 *                 
 *                 
 *                 
 * */ 
ddpg* init_ddpg(model* m1, model* m2, model* m3, model* m4, int batch_size, int threads, int regularization1,int regularization2, int m1_input,int m1_output,int m2_output,int m3_output,int gradient_descent_flag1,int gradient_descent_flag2, int buff_size, int max_frames, float lr1, float lr2, float momentum1, float momentum2, float lambda1, float lambda2, float tau,float epsilon_greedy, float lambda){
                    
    if(m1 == NULL || m2 == NULL || m3 == NULL || m4 == NULL){
        fprintf(stderr,"Error: you cannot have any model set to the null pointer!\n");
        exit(1);
    }
    
    if(m1->error_flag == NO_SET || m2->error_flag == NO_SET || m3->error_flag == NO_SET || m4->error_flag == NO_SET){
        fprintf(stderr,"Error: pls set the errors for m1,m2,m3,m4\n");
        exit(1);
    }
    
    int i;
    
    for(i = 0; i < m1->layers && m1->sla[i][0]; i++);

    if(i == m1->layers)
        i--;
    
    if(m1->sla[i][0] == RLS){
        fprintf(stderr,"Error: you cannot have residual layer as final layer in this library, look at the wiki why :(\n");
        exit(1);
    }
    
    for(i = 0; i < m2->layers && m2->sla[i][0]; i++);

    if(i == m2->layers)
        i--;
    
    if(m2->sla[i][0] == RLS){
        fprintf(stderr,"Error: you cannot have residual layer as final layer in this library, look at the wiki why :(\n");
        exit(1);
    }
    
    for(i = 0; i < m3->layers && m3->sla[i][0]; i++);

    if(i == m3->layers)
        i--;
    
    if(m3->sla[i][0] == RLS){
        fprintf(stderr,"Error: you cannot have residual layer as final layer in this library, look at the wiki why :(\n");
        exit(1);
    }
    
    for(i = 0; i < m4->layers && m4->sla[i][0]; i++);

    if(i == m4->layers)
        i--;
    
    if(m4->sla[i][0] == RLS){
        fprintf(stderr,"Error: you cannot have residual layer as final layer in this library, look at the wiki why :(\n");
        exit(1);
    }
    
    if(buff_size != m1_input){
        fprintf(stderr,"Error: buff size and m1_input should be the same!\n");
        exit(1);
    }
    
    ddpg* d = (ddpg*)malloc(sizeof(ddpg));
    d->m1 = m1;
    d->m2 = m2;
    d->m3 = m3;
    d->m4 = m4;
    d->batch_size = batch_size;
    d->regularization1 = regularization1;
    d->regularization2 = regularization2;
    d->n_weights1 = count_weights(m1);
    d->n_weights2 = count_weights(m2)+count_weights(m3)+count_weights(m4);
    d->index = 0;
    d->m1_input = m1_input;
    d->m1_output = m1_output;
    d->m2_output = m2_output;
    d->m3_output = m3_output;
    d->gradient_descent_flag1 = gradient_descent_flag1;
    d->gradient_descent_flag2 = gradient_descent_flag2;
    d->lr1 = lr1;
    d->lr2 = lr2;
    d->momentum1 = momentum1;
    d->momentum2 = momentum2;
    d->lambda1 = lambda1;
    d->lambda2 = lambda2;
    d->epsilon_greedy = epsilon_greedy;
    d->lambda = lambda;
    d->tau = tau;
    d->t1 = 1;
    d->t2 = 1;
    d->max_frames = max_frames;
    d->threads = threads;
    
    d->tm1 = (model**)malloc(sizeof(model*)*batch_size);
    d->tm2 = (model**)malloc(sizeof(model*)*batch_size);
    d->tm3 = (model**)malloc(sizeof(model*)*batch_size);
    d->tm4 = (model**)malloc(sizeof(model*)*batch_size);
    d->bm1 = (model**)malloc(sizeof(model*)*batch_size);
    d->bm2 = (model**)malloc(sizeof(model*)*batch_size);
    d->bm3 = (model**)malloc(sizeof(model*)*batch_size);
    d->bm4 = (model**)malloc(sizeof(model*)*batch_size);
    d->tm1_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->tm2_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->tm3_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->tm4_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->bm1_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->bm2_output_array = (float**)malloc(sizeof(float*)*batch_size);
    d->bm3_output_array = (float**)malloc(sizeof(float*)*batch_size);
    for(i = 0; i < batch_size; i++){
        d->tm1[i] = copy_model(m1);
        d->tm2[i] = copy_model(m2);
        d->tm3[i] = copy_model(m3);
        d->tm4[i] = copy_model(m4);
        d->bm1[i] = copy_model(m1);
        d->bm2[i] = copy_model(m2);
        d->bm3[i] = copy_model(m3);
        d->bm4[i] = copy_model(m4);
        d->tm1_output_array[i] = d->tm1[i]->output_layer;
        d->tm2_output_array[i] = d->tm2[i]->output_layer;
        d->tm3_output_array[i] = d->tm3[i]->output_layer;
        d->tm4_output_array[i] = d->tm4[i]->output_layer;
        d->bm1_output_array[i] = d->bm1[i]->output_layer;
        d->bm2_output_array[i] = d->bm2[i]->output_layer;
        d->bm3_output_array[i] = d->bm3[i]->output_layer;
    }
    
    d->buff1 = (float**)malloc(sizeof(float*)*max_frames);
    d->buff2 = (float**)malloc(sizeof(float*)*max_frames);
    d->rewards = (float*)malloc(sizeof(float)*max_frames);
    d->actions = (float**)malloc(sizeof(float*)*max_frames);
    d->terminal = (int*)malloc(sizeof(int)*max_frames);
    
    for(i = 0; i < max_frames; i++){
        d->buff1[i] = (float*)malloc(sizeof(float)*buff_size);
        d->buff2[i] = (float*)malloc(sizeof(float)*buff_size);
        d->actions[i] = (float*)malloc(sizeof(float)*m1_output);
    }
    
    d->buff_size = buff_size;
    
    return d;
    
}

/* This function deallocates the space allocated by a ddpg model
 * 
 * Inputs:
 * 
 *                 @ ddpg* d:= the model
 * */
void free_ddpg(ddpg* d){
    int i;
    for(i = 0; i < d->batch_size; i++){
        free_model(d->tm1[i]);
        free_model(d->tm2[i]);
        free_model(d->tm3[i]);
        free_model(d->tm4[i]);
        free_model(d->bm1[i]);
        free_model(d->bm2[i]);
        free_model(d->bm3[i]);
        free_model(d->bm4[i]);
    }
    free(d->tm1);
    free(d->tm2);
    free(d->tm3);
    free(d->tm4);
    free(d->bm1);
    free(d->bm2);
    free(d->bm3);
    free(d->bm4);
    free_model(d->m1);
    free_model(d->m2);
    free_model(d->m3);
    free_model(d->m4);
    
    for(i = 0; i < d->max_frames; i++){
        free(d->buff1[i]);
        free(d->buff2[i]);
        free(d->actions[i]);
    }
    free(d->buff1);
    free(d->buff2);
    free(d->actions);
    free(d->terminal);
    free(d->rewards);
    free(d);
}


/* This function computes the calculations that you can see in this pseudocode: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
 * from line 12 to line 16
 * 
 * Input:
 * 
 *             @ ddpg* d:= the ddpg model that i hope you have initialized
 * */
void ddpg_train(ddpg* d){
    
    int i;
    
    model_tensor_input_ff_multicore(d->tm1,1,1,d->m1_input,d->buff2,d->batch_size,d->threads);
    model_tensor_input_ff_multicore(d->tm2,1,1,d->m1_input,d->buff2,d->batch_size,d->threads);
    model_tensor_input_ff_multicore(d->tm3,1,1,d->m1_output,d->tm1_output_array,d->batch_size,d->threads);
    
    float** inputx3 = (float**)malloc(sizeof(float*)*d->batch_size);
    for(i = 0; i < d->batch_size; i++){
        inputx3[i] = (float*)malloc(sizeof(float)*d->m2_output*d->m3_output);
        copy_array(d->tm2_output_array[i],inputx3[i],d->m2_output);
        copy_array(d->tm3_output_array[i],&inputx3[i][d->m2_output],d->m3_output);
    }
    
    model_tensor_input_ff_multicore(d->tm4,1,1,d->m2_output+d->m3_output,inputx3,d->batch_size,d->threads);
    
    float** output = (float**)malloc(sizeof(float*)*d->batch_size);
    for(i = 0; i < d->batch_size; i++){
        reset_model(d->tm1[i]);
        reset_model(d->tm2[i]);
        reset_model(d->tm3[i]);
        reset_model(d->tm4[i]);
        output[i] = (float*)malloc(sizeof(float));
        output[i][0] = d->rewards[i]+d->lambda*(1-d->terminal[i])*d->tm4_output_array[i][0];
    }
    
    
    model_tensor_input_ff_multicore(d->bm2,1,1,d->m1_input,d->buff1,d->batch_size,d->threads);
    model_tensor_input_ff_multicore(d->bm3,1,1,d->m1_output,d->actions,d->batch_size,d->threads);
    for(i = 0; i < d->batch_size; i++){
        copy_array(d->bm2_output_array[i],inputx3[i],d->m2_output);
        copy_array(d->bm3_output_array[i],&inputx3[i][d->m2_output],d->m3_output);
    }
    
    
    float** ret_err = (float**)malloc(sizeof(float*)*d->batch_size);
    float** ret_err2 = (float**)malloc(sizeof(float*)*d->batch_size);
    float** ret_err3 = (float**)malloc(sizeof(float*)*d->batch_size);
    
    
    ff_error_bp_model_multicore(d->bm4,d->m2_output+d->m3_output,1,1,inputx3,d->batch_size,d->threads,output,ret_err);
    
    for(i = 0; i < d->batch_size; i++){
        ret_err3[i] = (float*)malloc(sizeof(float)*d->m3_output);
        copy_array(&ret_err[i][d->m2_output],ret_err3[i],d->m3_output);
    }
    
    model_tensor_input_bp_multicore(d->bm3,1,1,d->m1_output,d->actions,d->batch_size,d->threads,ret_err3,d->m3_output,ret_err2);
    model_tensor_input_bp_multicore(d->bm2,1,1,d->m1_input,d->buff1,d->batch_size,d->threads,ret_err,d->m2_output,ret_err2);
    
    sum_models_partial_derivatives(d->m2,d->bm2,d->batch_size);
    sum_models_partial_derivatives(d->m3,d->bm3,d->batch_size);
    sum_models_partial_derivatives(d->m4,d->bm4,d->batch_size);
    
    update_model(d->m4,d->lr1,d->momentum1,d->batch_size,d->gradient_descent_flag1,&d->m4->beta1_adam,&d->m4->beta2_adam,d->regularization1,d->n_weights1,d->lambda1,&d->t1);
    d->t1--;
    update_model(d->m3,d->lr1,d->momentum1,d->batch_size,d->gradient_descent_flag1,&d->m3->beta1_adam,&d->m3->beta2_adam,d->regularization1,d->n_weights1,d->lambda1,&d->t1);
    d->t1--;
    update_model(d->m2,d->lr1,d->momentum1,d->batch_size,d->gradient_descent_flag1,&d->m2->beta1_adam,&d->m2->beta2_adam,d->regularization1,d->n_weights1,d->lambda1,&d->t1);
    
    reset_model(d->m2);
    reset_model(d->m3);
    reset_model(d->m4);
    for(i = 0; i < d->batch_size; i++){
        reset_model(d->bm2[i]);
        reset_model(d->bm3[i]);
        reset_model(d->bm4[i]);
        paste_model(d->m2,d->bm2[i]);
        paste_model(d->m3,d->bm3[i]);
        paste_model(d->m4,d->bm4[i]);
    }
    
    model_tensor_input_ff_multicore(d->bm1,1,1,d->m1_input,d->buff1,d->batch_size,d->threads);
    model_tensor_input_ff_multicore(d->bm2,1,1,d->m1_input,d->buff1,d->batch_size,d->threads);
    model_tensor_input_ff_multicore(d->bm3,1,1,d->m1_output,d->bm1_output_array,d->batch_size,d->threads);
    for(i = 0; i < d->batch_size; i++){
        copy_array(d->bm2_output_array[i],inputx3[i],d->m2_output);
        copy_array(d->bm3_output_array[i],&inputx3[i][d->m2_output],d->m3_output);
    }
    model_tensor_input_ff_multicore(d->bm4,1,1,d->m2_output+d->m3_output,inputx3,d->batch_size,d->threads);
    for(i = 0; i < d->batch_size; i++){
        output[i][0] = -1;
    }
    
    model_tensor_input_bp_multicore(d->bm4,1,1,d->m2_output+d->m3_output,inputx3,d->batch_size,d->threads,output,1,ret_err);
    for(i = 0; i < d->batch_size; i++){
        copy_array(&ret_err[i][d->m2_output],ret_err3[i],d->m3_output);
    }
    model_tensor_input_bp_multicore(d->bm3,1,1,d->m1_output,d->bm1_output_array,d->batch_size,d->threads,ret_err3,d->m3_output,ret_err2);
    model_tensor_input_bp_multicore(d->bm1,1,1,d->m1_input,d->buff1,d->batch_size,d->threads,ret_err2,d->m1_output,ret_err);
    
    sum_models_partial_derivatives(d->m1,d->bm1,d->batch_size);
    update_model(d->m1,d->lr2,d->momentum2,d->batch_size,d->gradient_descent_flag2,&d->m1->beta1_adam,&d->m1->beta2_adam,d->regularization2,d->n_weights2,d->lambda2,&d->t2);
    
    reset_model(d->m1);
    for(i = 0; i < d->batch_size; i++){
        reset_model(d->bm1[i]);
        reset_model(d->bm2[i]);
        reset_model(d->bm3[i]);
        reset_model(d->bm4[i]);
        paste_model(d->m1,d->bm1[i]);
        slow_paste_model(d->m1,d->tm1[i],d->tau);
        slow_paste_model(d->m2,d->tm2[i],d->tau);
        slow_paste_model(d->m3,d->tm3[i],d->tau);
        slow_paste_model(d->m4,d->tm4[i],d->tau);
    }
    
    free_matrix(inputx3,d->batch_size);
    free_matrix(output,d->batch_size);
    free_matrix(ret_err3,d->batch_size);
    free(ret_err);
    free(ret_err2);
}
