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


void* recurrent_enc_dec_thread_ff(void* _args) {
    
    // depacking args
    thread_args_enc_dec_model* args = (thread_args_enc_dec_model*) _args;
    ff_recurrent_enc_dec(args->hidden_states,args->cell_states,args->input_model1,args->input_model2,args->m);
    return _args;
}

void* recurrent_enc_dec_thread_bp(void* _args) {
    
    // depacking args
    thread_args_enc_dec_model* args = (thread_args_enc_dec_model*) _args;
    if(args->ret_input_error1 != NULL && args->ret_input_error2 != NULL)
    args->returning_error[0] = bp_recurrent_enc_dec(args->hidden_states,args->cell_states,args->input_model1,args->input_model2,args->error_model,args->m,args->ret_input_error1[0],args->ret_input_error2[0]);
    else if(args->ret_input_error1 != NULL)
    args->returning_error[0] = bp_recurrent_enc_dec(args->hidden_states,args->cell_states,args->input_model1,args->input_model2,args->error_model,args->m,args->ret_input_error1[0],NULL);
    else if(args->ret_input_error2 != NULL)
    args->returning_error[0] = bp_recurrent_enc_dec(args->hidden_states,args->cell_states,args->input_model1,args->input_model2,args->error_model,args->m,NULL,args->ret_input_error2[0]);
    else
    args->returning_error[0] = bp_recurrent_enc_dec(args->hidden_states,args->cell_states,args->input_model1,args->input_model2,args->error_model,args->m,NULL,NULL);
    return _args;
}

/* This functions computes the feed forward of a rmodel for a batch of instances of the dataset
 * 
 * Inputs:
 * 
 *             
 *             @ float*** hidden_states:= the hidden states of each instance, dimensions: mini_batch_size*m[0]->window*m[0]->size
 *             @ float*** cell_states:= the cell states of each instance,  dimensions: mini_batch_size*m[0]->window*m[0]->size
 *             @ float*** input_model1:= the inputs of each instance of the batch,  dimensions: mini_batch_size*m[0]->window*m[0]->size for encoder
 *             @ float*** input_model2:= the inputs of each instance of the batch,  dimensions: mini_batch_size*m[0]->window*m[0]->size for decoder
 *             @ recurrent_enc_dec** m:= the models of the batch,  dimensions: mini_batch_size*1
 *             @ int mini_batch_size:= the batch size used
 *             @ int threads:= the number of threads you want to use
 * 
 * */
void ff_recurrent_enc_dec_multicore(float*** hidden_states, float*** cell_states, float*** input_model1, float*** input_model2, recurrent_enc_dec** m, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_enc_dec_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_enc_dec_model*)malloc(sizeof(thread_args_enc_dec_model));
            args[j]->m = m[i+j];
            args[j]->hidden_states = hidden_states[i+j];
            args[j]->cell_states = cell_states[i+j];
            args[j]->input_model1 = input_model1[i+j];
            args[j]->input_model2 = input_model2[i+j];
            pthread_create(thread+j, NULL, recurrent_enc_dec_thread_ff, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

/* This functions computes the back propagation of a rmodel for a batch of instances of the dataset
 * 
 * Inputs:
 * 
 *             
 *             @ float*** hidden_states:= the hidden states of each instance, dimensions: mini_batch_size*m[0]->window*m[0]->size
 *             @ float*** cell_states:= the cell states of each instance,  dimensions: mini_batch_size*m[0]->window*m[0]->size
 *             @ float*** input_model1:= the inputs of each instance of the batch,  dimensions: mini_batch_size*m[0]->window*m[0]->size for encoder
 *             @ float*** input_model2:= the inputs of each instance of the batch,  dimensions: mini_batch_size*m[0]->window*m[0]->size for decoder
 *             @ recurrent_enc_dec** m:= the models of the batch,  dimensions: mini_batch_size*1
 *             @ float*** error_model:= the errors of each instance of the batch, dimensions: mini_batch_size*window*m[0]->size
 *             @ int mini_batch_size:= the batch size used
 *             @ int threads:= the number of threads you want to use
 *             @ float**** returning_error:= where will be stored the errors, dimensions:= mini_batch_size*m[0]->layers*4*m[0]->size, must be allocated only the batch size
 *             @ float*** returning_input_error1:= where will be stored the errors of the input of the model, dimensions:= mini_batch_size*m[0]->window*m[0]->size for encoder
 *             @ float*** returning_input_error2:= where will be stored the errors of the input of the model, dimensions:= mini_batch_size*m[0]->window*m[0]->size for decoder
 * 
 * */
void bp_recurrent_enc_dec_multicore(float*** hidden_states, float*** cell_states, float*** input_model1,float*** input_model2, recurrent_enc_dec** m, float*** error_model, int mini_batch_size, int threads, float**** returning_error, float*** returning_input_error1,float*** returning_input_error2){
    pthread_t thread[threads];
    thread_args_enc_dec_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_enc_dec_model*)malloc(sizeof(thread_args_enc_dec_model));
            args[j]->m = m[i+j];
            args[j]->hidden_states = hidden_states[i+j];
            args[j]->cell_states = cell_states[i+j];
            args[j]->input_model1 = input_model1[i+j];
            args[j]->input_model2 = input_model2[i+j];
            args[j]->error_model = error_model[i+j];
            args[j]->returning_error = &returning_error[i+j];
            if(returning_input_error1 != NULL)
            args[j]->ret_input_error1 = &returning_input_error1[i+j];
            else
            args[j]->ret_input_error1 = NULL;
            if(returning_input_error2 != NULL)
            args[j]->ret_input_error2 = &returning_input_error2[i+j];
            else
            args[j]->ret_input_error2 = NULL;
            pthread_create(thread+j, NULL, recurrent_enc_dec_thread_bp, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}
