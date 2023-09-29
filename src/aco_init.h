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

#ifndef __ACO_INIT_H__
#define __ACO_INIT_H__

params* init_params(int size, int input_size, int dimension1, int dimension2, int dimension3);
activation* init_activation(int activation_flag);
fcl_func* init_fcl_func(int input, int output);
aco_edge* init_aco_edge(aco_node* input, aco_node* output, int operation_flag);
aco_node* init_aco_node(params* weights, params* biases, activation* a, fcl_func* f);
void free_params(params* p);
void free_fcl_func(fcl_func* f);
void free_aco_node(aco_node* n);
void reset_fcl_func(fcl_func* f);
void reset_aco_edge(aco_edge* e);
void reset_aco_node(aco_node* n);
void add_aco_edge(aco_node* n, aco_edge* e, int input_edge);
int node_state(aco_node* n);
aco_tracker* init_aco_tracker();
void reset_aco_tracker(aco_tracker* a, int free_arrays, int free_model_flag);
void copy_param(params* original, params* copy);


#endif
