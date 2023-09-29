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


void* model_thread_ff(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    model_tensor_input_ff(args->m,args->channels,args->rows,args->cols,args->input);
    return _args;
}

void* model_thread_sum(void* _args) {
    
    // depacking args
    thread_args_models* args = (thread_args_models*) _args;
    sum_models_partial_derivatives_multithread(args->m, args->m[0], args->n, args->depth);
    return _args;
}

void* model_thread_ff_opt(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    model_tensor_input_ff_without_learning_parameters(args->m,args->real_m,args->channels,args->rows,args->cols,args->input);
    return _args;
}

void* model_thread_ff_loss_reset_only_for_ff_opt(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    int i;
    double error = 0;
    for(i = 0; i < args->dataset_size; i++){
        model_tensor_input_ff_without_learning_parameters(args->m,args->real_m,args->channels,args->rows,args->cols,args->dataset_input[i]);
        compute_model_error_only_for_ff(args->m, args->dataset_output[i]);
        error+=sum_over_input(args->m->error, args->m->output_dimension);
        reset_model_only_for_ff(args->m);
    }
    
    args->only_for_ff_ret[0] = error;
    
    
    return _args;
}

void* model_thread_bp(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    if(args->returning_error != NULL)
    args->returning_error[0] = model_tensor_input_bp(args->m,args->channels,args->rows,args->cols,args->input,args->error,args->error_dimension);
    else
    model_tensor_input_bp(args->m,args->channels,args->rows,args->cols,args->input,args->error,args->error_dimension);
    return _args;
}

void* model_thread_bp_opt(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    if(args->returning_error != NULL)
    args->returning_error[0] = model_tensor_input_bp_without_learning_parameters(args->m,args->real_m,args->channels,args->rows,args->cols,args->input,args->error,args->error_dimension);
    else
    model_tensor_input_bp_without_learning_parameters(args->m,args->real_m,args->channels,args->rows,args->cols,args->input,args->error,args->error_dimension);
    return _args;
}

void* model_thread_ff_bp(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    if(args->returning_error != NULL)
    args->returning_error[0] = ff_error_bp_model_once(args->m,args->channels,args->rows,args->cols,args->input,args->error);
    else
    ff_error_bp_model_once(args->m, args->channels,args->rows,args->cols,args->input,args->error);
    return _args;
}

void* model_thread_ff_bp_opt(void* _args) {
    
    // depacking args
    thread_args_model* args = (thread_args_model*) _args;
    if(args->returning_error != NULL)
    args->returning_error[0] = ff_error_bp_model_once_opt(args->m,args->real_m,args->channels,args->rows,args->cols,args->input,args->error);
    else
    ff_error_bp_model_once_opt(args->m,args->real_m,args->channels,args->rows,args->cols,args->input,args->error);
    return _args;
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
void model_tensor_input_ff_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads){
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
            args[j]->real_m = m2;
            pthread_create(thread+j, NULL, model_thread_ff_opt, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

double model_tensor_input_ff_multicore_only_for_ff_loss_reset_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, float** outputs, int mini_batch_size, int threads){
    if(mini_batch_size < threads)
        threads = mini_batch_size;
    pthread_t thread[threads];
    thread_args_model* args[threads];
    
    int j;
    int value = mini_batch_size/threads;
    double ret = 0;
    for(j = 0; j < threads; j++){
        args[j] = (thread_args_model*)malloc(sizeof(thread_args_model));
        args[j]->m = m[j];
        args[j]->channels = depth;
        args[j]->rows = rows;
        args[j]->cols = cols;
        args[j]->dataset_input = &inputs[j*value];
        args[j]->dataset_output = &outputs[j*value];
        args[j]->dataset_size = value;
        if(mini_batch_size - j*value < value)
            args[j]->dataset_size = mini_batch_size - j*value;
        args[j]->real_m = m2;
        args[j]->only_for_ff_ret = (double*)calloc(1,sizeof(double));
        pthread_create(thread+j, NULL, model_thread_ff_loss_reset_only_for_ff_opt, args[j]);
        
    }
                
    for(j = 0; j < threads; j++) {
        pthread_join(thread[j], NULL);
        ret += args[j]->only_for_ff_ret[0];
        free(args[j]->only_for_ff_ret);
        free(args[j]);
    }
    return ret;
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
            if(returning_error == NULL)
            args[j]->returning_error = NULL;
            else
            args[j]->returning_error = &returning_error[i+j];            
            pthread_create(thread+j, NULL, model_thread_bp, args[j]);
            
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
void model_tensor_input_bp_multicore_opt(model** m,model*m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error){
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
            args[j]->real_m = m2;
            if(returning_error == NULL)
            args[j]->returning_error = NULL;
            else
            args[j]->returning_error = &returning_error[i+j];            
            pthread_create(thread+j, NULL, model_thread_bp_opt, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

/* This functions computes the feed forward and back propagation of a model for a batch of instances of the dataset
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
 *             @ float** outputs:= the outputs of the batch, dimensions: mini_batch_size*error_dimension
 *             @ float** returning_error:= where is stored the error of these models, dimensions: mini_batch_size*(depth*rows*cols)
 *             
 * 
 * */
void ff_error_bp_model_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error){
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
            args[j]->error = outputs[i+j];
            if(returning_error != NULL)
                args[j]->returning_error = &returning_error[i+j];
            else
                args[j]->returning_error = NULL;
            pthread_create(thread+j, NULL, model_thread_ff_bp, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }
}
/* This functions computes the feed forward and back propagation of a model for a batch of instances of the dataset
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
 *             @ float** outputs:= the outputs of the batch, dimensions: mini_batch_size*error_dimension
 *             @ float** returning_error:= where is stored the error of these models, dimensions: mini_batch_size*(depth*rows*cols)
 *             
 * 
 * */
void ff_error_bp_model_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error){
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
            args[j]->error = outputs[i+j];
            args[j]->real_m = m2;
            if(returning_error != NULL)
                args[j]->returning_error = &returning_error[i+j];
            else
                args[j]->returning_error = NULL;
            pthread_create(thread+j, NULL, model_thread_ff_bp_opt, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }
}




model* sum_models_partial_derivatives_multithread(model** batch_m, model* m, int n, int depth){
    if(depth == 0 && n <=3){
        sum_models_partial_derivatives(m,batch_m,n);
        return m;
    }
    
    if(n == 0)
        return NULL;
    else if(n == 1)
        return batch_m[0];
    else if(n == 2){
        sum_model_partial_derivatives(batch_m[0],batch_m[1],batch_m[0]);
        return batch_m[0];
    }
    else if(n == 3){
        sum_models_partial_derivatives(batch_m[0],&batch_m[1],2);
        return batch_m[0];
    }
    pthread_t thread[2];
    thread_args_models* args[2];
    int size = (int)(n/2);
    args[0] = (thread_args_models*)malloc(sizeof(thread_args_models));
    args[0]->depth = depth+1;
    args[0]->n = size;
    args[0]->m = batch_m;
    args[1] = (thread_args_models*)malloc(sizeof(thread_args_models));
    args[1]->depth = depth+1;
    args[1]->n = n-size;
    args[1]->m = batch_m+size;
    pthread_create(thread, NULL, model_thread_sum, args[0]);
    pthread_create(thread+1, NULL, model_thread_sum, args[1]);
    pthread_join(thread[0],NULL);
    pthread_join(thread[1],NULL);
    free(args[0]);
    free(args[1]);
    model* t1 = batch_m[0];
    model* t2 = batch_m[size];
    sum_model_partial_derivatives(t1,t2,t1);
    if(depth == 0){
        sum_model_partial_derivatives(t1,m,m);
        t1 = m;
    }
    
    return t1;
}
