#include "llab.h"


void* rmodel_thread_ff(void* _args) {
    
    // depacking args
    thread_args_rmodel* args = (thread_args_rmodel*) _args;
    ff_rmodel_lstm(args->hidden_states,args->cell_states,args->input_model,args->m);
    
}

void* rmodel_thread_bp(void* _args) {
    
    // depacking args
    thread_args_rmodel* args = (thread_args_rmodel*) _args;
    args->returning_error[0] = bp_rmodel_lstm(args->hidden_states,args->cell_states,args->input_model,args->error_model,args->m);
}

void ff_rmodel_lstm_multicore(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_rmodel* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_rmodel*)malloc(sizeof(thread_args_rmodel));
            args[j]->m = m[i+j];
            args[j]->hidden_states = hidden_states[i+j];
            args[j]->cell_states = cell_states[i+j];
            args[j]->input_model = input_model[i+j];
            pthread_create(thread+j, NULL, rmodel_thread_ff, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

void bp_rmodel_lstm_multicore(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, float*** error_model, int mini_batch_size, int threads, float**** returning_error){
    pthread_t thread[threads];
    thread_args_rmodel* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_rmodel*)malloc(sizeof(thread_args_rmodel));
            args[j]->m = m[i+j];
            args[j]->hidden_states = hidden_states[i+j];
            args[j]->cell_states = cell_states[i+j];
            args[j]->input_model = input_model[i+j];
            args[j]->error_model = error_model[i+j];
            args[j]->returning_error = &returning_error[i+j];
            pthread_create(thread+j, NULL, rmodel_thread_bp, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}
