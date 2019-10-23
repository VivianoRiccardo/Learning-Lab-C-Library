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


void* gan_model_thread_ff_disc(void* _args) {
    
    // depacking args
    thread_args_gan_model* args = (thread_args_gan_model*) _args;
    discriminator_feed_forward(args->gm,args->real_input,args->noise_input,args->g_d_in,args->g_i_in,args->g_j_in,args->d_d_in,args->d_i_in,args->d_j_in);
    
    return _args;
}

void* gan_model_thread_ff_gen(void* _args) {
    
    // depacking args
    thread_args_gan_model* args = (thread_args_gan_model*) _args;
    generator_feed_forward(args->gm,args->noise_input,args->g_d_in,args->g_i_in,args->g_j_in,args->d_d_in,args->d_i_in,args->d_j_in);
    return _args;
}


void* gan_model_thread_bp_disc(void* _args) {
    
    // depacking args
    thread_args_gan_model* args = (thread_args_gan_model*) _args;
    discriminator_back_propagation(args->gm,args->real_input,args->noise_input,args->g_d_in,args->g_i_in,args->g_j_in,args->d_d_in,args->d_i_in,args->d_j_in);
    return _args;
}


void* gan_model_thread_bp_gen(void* _args) {
    
    // depacking args
    thread_args_gan_model* args = (thread_args_gan_model*) _args;
    args->ret_err[0] = generator_back_propagation(args->gm,args->noise_input,args->g_d_in,args->g_i_in,args->g_j_in,args->d_d_in,args->d_i_in,args->d_j_in,args->output_size);
    return _args;
}

void discriminator_tensor_input_ff_multicore(ganmodel** m,float** real_input, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_gan_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_gan_model*)malloc(sizeof(thread_args_gan_model));
            args[j]->gm = m[i+j];
            args[j]->g_d_in = tensor_input_g_depth;
            args[j]->g_i_in = tensor_input_g_i;
            args[j]->g_j_in = tensor_input_g_j;
            args[j]->d_d_in = tensor_input_d_depth;
            args[j]->d_i_in = tensor_input_d_i;
            args[j]->d_j_in = tensor_input_d_j;
            args[j]->output_size = tensor_input_d_depth*tensor_input_d_i*tensor_input_d_j;
            args[j]->real_input = real_input[i+j];
            args[j]->noise_input = noise_input[i+j];
            pthread_create(thread+j, NULL, gan_model_thread_ff_disc, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

void generator_tensor_input_ff_multicore(ganmodel** m, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_gan_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_gan_model*)malloc(sizeof(thread_args_gan_model));
            args[j]->gm = m[i+j];
            args[j]->g_d_in = tensor_input_g_depth;
            args[j]->g_i_in = tensor_input_g_i;
            args[j]->g_j_in = tensor_input_g_j;
            args[j]->d_d_in = tensor_input_d_depth;
            args[j]->d_i_in = tensor_input_d_i;
            args[j]->d_j_in = tensor_input_d_j;
            args[j]->output_size = tensor_input_d_depth*tensor_input_d_i*tensor_input_d_j;
            args[j]->real_input = NULL;
            args[j]->noise_input = noise_input[i+j];
            pthread_create(thread+j, NULL, gan_model_thread_ff_gen, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }

}

void discriminator_tensor_input_bp_multicore(ganmodel** m,float** real_input, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads){
    pthread_t thread[threads];
    thread_args_gan_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_gan_model*)malloc(sizeof(thread_args_gan_model));
            args[j]->gm = m[i+j];
            args[j]->g_d_in = tensor_input_g_depth;
            args[j]->g_i_in = tensor_input_g_i;
            args[j]->g_j_in = tensor_input_g_j;
            args[j]->d_d_in = tensor_input_d_depth;
            args[j]->d_i_in = tensor_input_d_i;
            args[j]->d_j_in = tensor_input_d_j;
            args[j]->output_size = tensor_input_d_depth*tensor_input_d_i*tensor_input_d_j;
            args[j]->real_input = real_input[i+j];
            args[j]->noise_input = noise_input[i+j];
            args[j]->ret_err = NULL;
            pthread_create(thread+j, NULL, gan_model_thread_bp_disc, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }
}

void generator_tensor_input_bp_multicore(ganmodel** m, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads, float** returning_error){
    pthread_t thread[threads];
    thread_args_gan_model* args[threads];
    
    int i,j;
    
    for(i = 0; i < mini_batch_size; i+=threads){
        for(j = 0; j < threads && j+i < mini_batch_size; j++){
            args[j] = (thread_args_gan_model*)malloc(sizeof(thread_args_gan_model));
            args[j]->gm = m[i+j];
            args[j]->g_d_in = tensor_input_g_depth;
            args[j]->g_i_in = tensor_input_g_i;
            args[j]->g_j_in = tensor_input_g_j;
            args[j]->d_d_in = tensor_input_d_depth;
            args[j]->d_i_in = tensor_input_d_i;
            args[j]->d_j_in = tensor_input_d_j;
            args[j]->output_size = tensor_input_d_depth*tensor_input_d_i*tensor_input_d_j;
            args[j]->noise_input = noise_input[i+j];
            args[j]->ret_err = &returning_error[i+j];
            pthread_create(thread+j, NULL, gan_model_thread_bp_gen, args[j]);
            
            }
                
        for(j = 0; j < threads && j+i < mini_batch_size; j++) {
            pthread_join(thread[j], NULL);
            free(args[j]);
        }
    }
}

