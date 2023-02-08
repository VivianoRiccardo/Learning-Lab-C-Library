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
#include "genome.h"

#ifndef __MULTI_CORE_NEAT_H__
#define __MULTI_CORE_NEAT_H__

void* genome_thread_ff(void* _args);
float** feed_forward_multi_thread(int threads, float** inputs,genome** g, int global_inn_numb_nodes, int global_inn_numb_connections);
void* genome_thread_ff_opt(void* _args);
float** feed_forward_multi_thread_opt(int number_of_genomes, int threads, float** inputs,genome** g, int global_inn_numb_nodes, int global_inn_numb_connections);
float** feed_forward_multi_thread_opt_with_indices(int number_of_genomes, int threads, float** inputs,genome** g, int* indices, int global_inn_numb_nodes, int global_inn_numb_connections);

#endif
