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
