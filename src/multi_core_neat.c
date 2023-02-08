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

void* genome_thread_ff(void* _args) {
    
    // depacking args
    thread_args_genome* args = (thread_args_genome*) _args;
    args->output[args->index] = feed_forward(args->g,args->input,args->global_inn_numb_nodes,args->global_inn_numb_connections);
    return _args;
}

void* genome_thread_ff_opt(void* _args) {
    
    // depacking args
    thread_args_genome_opt* args = (thread_args_genome_opt*) _args;
    if(!args->number_of_genomes){
        args->output[args->index] = NULL;
        return _args;
    }
    int i;
    for(i = 0; i < args->number_of_genomes; i++){
        args->output[args->index+i] = feed_forward(args->g[i],args->input[i],args->global_inn_numb_nodes,args->global_inn_numb_connections);
    }
    
    return _args;
}


float** feed_forward_multi_thread(int threads, float** inputs,genome** g, int global_inn_numb_nodes, int global_inn_numb_connections){
    pthread_t thread[threads];
    thread_args_genome* args[threads];
    int i;
    float** output = (float**)malloc(sizeof(float*)*threads);
    for(i = 0; i < threads; i++){
        args[i] = (thread_args_genome*)malloc(sizeof(thread_args_genome));
        args[i]->g = g[i];
        args[i]->output = output;
        args[i]->index = i;
        args[i]->input= inputs[i];
        args[i]->global_inn_numb_connections = global_inn_numb_connections;
        args[i]->global_inn_numb_nodes = global_inn_numb_nodes;
        pthread_create(thread+i,NULL,genome_thread_ff,args[i]);
    }
    
    for(i = 0; i < threads; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    return output;
}

float** feed_forward_multi_thread_opt(int number_of_genomes, int threads, float** inputs,genome** g, int global_inn_numb_nodes, int global_inn_numb_connections){
    if(threads == 0 || number_of_genomes == 0){
        return NULL;
    }
    if(threads > number_of_genomes){
        threads = number_of_genomes;
    }
    pthread_t thread[threads];
    thread_args_genome_opt* args[threads];
    int i;
    int assigned_genomes = number_of_genomes/threads;
    float** output = (float**)malloc(sizeof(float*)*number_of_genomes);
    for(i = 0; i < threads; i++){
        args[i] = (thread_args_genome_opt*)malloc(sizeof(thread_args_genome_opt));
        if(i == threads -1){
            args[i]->number_of_genomes = number_of_genomes-i*assigned_genomes;
        }
        else{
            args[i]->number_of_genomes = assigned_genomes;
        }
        args[i]->g = &g[i*assigned_genomes];
        args[i]->output = output;
        args[i]->index = i*assigned_genomes;
        args[i]->input= &inputs[i*assigned_genomes];
        args[i]->global_inn_numb_connections = global_inn_numb_connections;
        args[i]->global_inn_numb_nodes = global_inn_numb_nodes;
        pthread_create(thread+i,NULL,genome_thread_ff_opt,args[i]);
    }
    
    for(i = 0; i < threads; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    return output;
}


float** feed_forward_multi_thread_opt_with_indices(int number_of_genomes, int threads, float** inputs,genome** g, int* indices, int global_inn_numb_nodes, int global_inn_numb_connections){
    if(threads == 0 || number_of_genomes == 0){
        return NULL;
    }
    if(threads > number_of_genomes){
        threads = number_of_genomes;
    }
    pthread_t thread[threads];
    thread_args_genome_opt* args[threads];
    int i,j;
    int n_real_genomes = 0;
    for(i = 0; i < number_of_genomes; i++){
        if(indices[i])
            n_real_genomes++;
    }
    if(!n_real_genomes)
        return NULL;
    if(threads > n_real_genomes)
        threads = n_real_genomes;
    int assigned_genomes = n_real_genomes/threads;
    genome** real_g = (genome**)malloc(sizeof(genome*)*n_real_genomes);
    float** real_inputs = (float**)malloc(sizeof(float*)*n_real_genomes);
    for(i = 0,j=0; i < number_of_genomes; i++){
        if(indices[i]){
            real_g[j] = g[i];
            real_inputs[j] = inputs[i];
            j++;
        }
    } 
    float** output = (float**)malloc(sizeof(float*)*n_real_genomes);
    for(i = 0; i < threads; i++){
        args[i] = (thread_args_genome_opt*)malloc(sizeof(thread_args_genome_opt));
        if(i == threads -1){
            args[i]->number_of_genomes = n_real_genomes-i*assigned_genomes;
        }
        else{
            args[i]->number_of_genomes = assigned_genomes;
        }
        args[i]->g = &real_g[i*assigned_genomes];
        args[i]->output = output;
        args[i]->index = i*assigned_genomes;
        args[i]->input= &real_inputs[i*assigned_genomes];
        args[i]->global_inn_numb_connections = global_inn_numb_connections;
        args[i]->global_inn_numb_nodes = global_inn_numb_nodes;
        pthread_create(thread+i,NULL,genome_thread_ff_opt,args[i]);
    }
    
    for(i = 0; i < threads; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    free(real_g);
    free(real_inputs);
    return output;
}
