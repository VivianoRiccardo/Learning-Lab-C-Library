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

#ifndef __FULLY_CONNECTED_LAYERS_H__
#define __FULLY_CONNECTED_LAYERS_H__

fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold);
void free_fully_connected(fcl* f);
void save_fcl(fcl* f, int n);
void copy_fcl_params(fcl* f, float* weights, float* biases);
fcl* load_fcl(FILE* fr);
fcl* copy_fcl(fcl* f);
void paste_fcl(fcl* f, fcl* copy);
fcl* reset_fcl(fcl* f);
unsigned long long int size_of_fcls(fcl* f);
void slow_paste_fcl(fcl* f,fcl* copy, float tau);
fcl* fcl_merge(fcl* f1, fcl* f2);
int get_array_size_params(fcl* f);
void memcopy_params_to_vector(fcl* f, float* vector);
void memcopy_vector_to_params(fcl* f, float* vector);
void memcopy_derivative_params_to_vector(fcl* f, float* vector);
void memcopy_vector_to_derivative_params(fcl* f, float* vector);
void set_fully_connected_biases_to_zero(fcl* f);
void set_fully_connected_unused_weights_to_zero(fcl* f);
int fcl_adjusting_weights_after_edge_popup(fcl* f, int* used_input, int* used_output, int layer_flag, int input_size);
int* get_used_inputs(fcl* f, int* used_input, int flag, int input_size);
int* get_used_outputs(fcl* f, int* used_output, int flag, int output_size);
#endif
