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

#ifndef __ACO_BUILDING_H__
#define __ACO_BUILDING_H__

aco_node** aco_create_activations_node();
void aco_attach_activations_to_fcl(aco_node* f, aco_node** a);
aco_node* aco_attach_fcl_to_params(aco_node** n, int size, int input_size, int output_size);
aco_node** aco_build_layer_param_nodes(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1);
aco_node** get_nodes_to_attach(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1, int activations);
void aco_add_output_nodes_to_node(aco_node* n, aco_node** ns, int size, int operation_flag);
int attach_params_to_root(aco_node* root, aco_node** ns, int size);
aco_node** attach_layer(aco_node* root, int input_size, int output_size, int width, int depth, int sub_matrix_dimension1, int activations);
aco_node* aco_create_activation_node(int activation_flag);
aco_node** aco_build_layer_param_nodes_complete(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1);
aco_node** aco_build_layer_param_nodes_complete2(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1);



#endif
