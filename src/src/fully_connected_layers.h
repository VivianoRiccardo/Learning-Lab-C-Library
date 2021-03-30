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

fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag);
void free_fully_connected(fcl* f);
void save_fcl(fcl* f, int n);
void copy_fcl_params(fcl* f, float* weights, float* biases);
fcl* load_fcl(FILE* fr);
fcl* copy_fcl(fcl* f);
void paste_fcl(fcl* f, fcl* copy);
fcl* reset_fcl(fcl* f);
unsigned long long int size_of_fcls(fcl* f);
void slow_paste_fcl(fcl* f,fcl* copy, float tau);
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
void heavy_save_fcl(fcl* f, int n);
fcl* heavy_load_fcl(FILE* fr);
void sum_score_fcl(fcl* input1, fcl* input2, fcl* output);
void dividing_score_fcl(fcl* f, float value);
void set_fcl_only_dropout(fcl* f);
void reset_score_fcl(fcl* f);
void reinitialize_scores_fcl(fcl* f, float percentage, float goodness);
void free_fully_connected_for_edge_popup(fcl* f);
fcl* light_load_fcl(FILE* fr);
fcl* light_reset_fcl(fcl* f);
int get_array_size_weights(fcl* f);
void memcopy_scores_to_vector(fcl* f, float* vector);
void memcopy_vector_to_scores(fcl* f, float* vector);
fcl* copy_light_fcl(fcl* f);
fcl* reset_fcl_for_edge_popup(fcl* f);
fcl* reset_fcl_without_dwdb(fcl* f);
void paste_fcl_for_edge_popup(fcl* f,fcl* copy);
void free_fully_connected_complementary_edge_popup(fcl* f);
void memcopy_weights_to_vector(fcl* f, float* vector);
void memcopy_vector_to_weights(fcl* f, float* vector);
void compare_score_fcl(fcl* input1, fcl* input2, fcl* output);
fcl* reset_edge_popup_d_fcl(fcl* f);
void set_low_score_fcl(fcl* f);
fcl* reset_fcl_except_partial_derivatives(fcl* f);
void reinitialize_w_fcl(fcl* f);

#endif
