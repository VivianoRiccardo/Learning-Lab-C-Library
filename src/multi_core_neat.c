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
