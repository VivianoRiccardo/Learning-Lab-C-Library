/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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


void* model_thread_ff(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    model_tensor_input_ff(args->m,args->channels,args->rows,args->cols,args->input);
    
}

void* model_thread_bp(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    args->returning_error[0] = model_tensor_input_bp(args->m,args->channels,args->rows,args->cols,args->input,args->error,args->error_dimension);
}


/* This functions computes the feed forward of a model for a batch of instances of the dataset
 * 
 * Inputs:
 * 
 *             
 *             @ model** m:= the models of the batch, dimensions: mini_batch_size*1
 *             @ int depth:= the depth of the input tensor
 *             @ int rows:= the number of rows of the input tensor
 *             @ int cols:= the number of columns of the input tensor
 *             @ float** inputs:= the inputs of the batch, dimensions: mini_batch_size*(depth*rows*cols)
 *             @ int mini_batch_size:= the size of the batch
 *             @ int threads:= the number of threads you want to use
 *             
 * 
 * */
void model_tensor_input_ff_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_model*)malloc(sizeof(thread_args_model));
            args[j]->m = m[i+j];
            args[j]->channels = depth;
            args[j]->rows = rows;
            args[j]->cols = cols;
            args[j]->input = inputs[i+j];
            pthread_create(thread+j, NULL, model_thread_ff, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

/* This functions computes the back propagation of a model for a batch of instances of the dataset
 * 
 * Inputs:
 * 
 *             
 *             @ model** m:= the models of the batch, dimensions: mini_batch_size*1
 *             @ int depth:= the depth of the input tensor
 *             @ int rows:= the number of rows of the input tensor
 *             @ int cols:= the number of columns of the input tensor
 *             @ float** inputs:= the inputs of the batch, dimensions: mini_batch_size*(depth*rows*cols)
 *             @ int mini_batch_size:= the size of the batch
 *             @ int threads:= the number of threads you want to use
 *             @ float** errors:= the errors of the batch, dimensions: mini_batch_size*error_dimension
 *             @ int error_dimension:= the dimension of each error
 *             @ float** returning_error:= where is stored the error of these models, dimensions: mini_batch_size*(depth*rows*cols)
 *             
 * 
 * */
void model_tensor_input_bp_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error){
    pthread_t thread[threads];
    thread_args_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_model*)malloc(sizeof(thread_args_model));
            args[j]->m = m[i+j];
            args[j]->channels = depth;
            args[j]->rows = rows;
            args[j]->cols = cols;
            args[j]->input = inputs[i+j];
            args[j]->error = errors[i+j];
            args[j]->error_dimension = error_dimension;
            args[j]->returning_error = &returning_error[i+j];
            pthread_create(thread+j, NULL, model_thread_bp, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

} 
