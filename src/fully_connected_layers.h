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

fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode);
int exists_params_fcl(fcl* f);
int exists_d_params_fcl(fcl* f);
int exists_dropout_stuff_fcl(fcl* f);
int exists_edge_popup_stuff_fcl(fcl* f);
int exists_activation_fcl(fcl* f);
int exists_normalization_fcl(fcl* f);
void free_fully_connected(fcl* f);
void free_fully_connected_for_edge_popup(fcl* f);
void free_fully_connected_complementary_edge_popup(fcl* f);
void save_fcl(fcl* f, int n);
void copy_fcl_params(fcl* f, float* weights,float* noisy_weights,float* noisy_biases, float* biases);
fcl* load_fcl(FILE* fr);
fcl* copy_fcl(fcl* f);
fcl* copy_light_fcl(fcl* f);
fcl* reset_fcl(fcl* f);
fcl* reset_fcl_except_partial_derivatives(fcl* f);
fcl* reset_fcl_without_dwdb(fcl* f);
fcl* reset_fcl_for_edge_popup(fcl* f);
uint64_t size_of_fcls(fcl* f);
void paste_fcl(fcl* f,fcl* copy);
void paste_w_fcl(fcl* f,fcl* copy);
void slow_paste_fcl(fcl* f,fcl* copy, float tau);
uint64_t get_array_size_params(fcl* f);
uint64_t get_array_size_scores_fcl(fcl* f);
uint64_t get_array_size_weights(fcl* f);
void memcopy_vector_to_params(fcl* f, float* vector);
void memcopy_vector_to_scores(fcl* f, float* vector);
void memcopy_params_to_vector(fcl* f, float* vector);
void memcopy_weights_to_vector(fcl* f, float* vector);
void memcopy_vector_to_weights(fcl* f, float* vector);
void memcopy_scores_to_vector(fcl* f, float* vector);
void memcopy_vector_to_derivative_params(fcl* f, float* vector);
void memcopy_derivative_params_to_vector(fcl* f, float* vector);
void set_fully_connected_biases_to_zero(fcl* f);
void set_fully_connected_unused_weights_to_zero(fcl* f);
void sum_score_fcl(fcl* input1, fcl* input2, fcl* output);
void compare_score_fcl(fcl* input1, fcl* input2, fcl* output);
void compare_score_fcl_with_vector(fcl* input1, float* input2, fcl* output);
void dividing_score_fcl(fcl* f, float value);
void set_fcl_only_dropout(fcl* f);
void reset_score_fcl(fcl* f);
void reinitialize_weights_according_to_scores_fcl(fcl* f, float percentage, float goodness);
void reinitialize_w_fcl(fcl* f);
fcl* reset_edge_popup_d_fcl(fcl* f);
void set_low_score_fcl(fcl* f);
int* get_used_outputs(fcl* f, int* used_output, int flag, int output_size);
fcl* copy_fcl_without_learning_parameters(fcl* f);
fcl* fully_connected_without_learning_parameters(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode);
fcl* reset_fcl_without_learning_parameters(fcl* f);
uint64_t size_of_fcls_without_learning_parameters(fcl* f);
void paste_fcl_without_learning_parameters(fcl* f,fcl* copy);
fcl* reset_fcl_without_dwdb_without_learning_parameters(fcl* f);
uint64_t count_weights_fcl(fcl* f);
void make_the_fcl_only_for_ff(fcl* f);
fcl* fully_connected_without_arrays(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode);
void free_fully_connected_without_arrays(fcl* f);
void inference_fcl(fcl* f);
void train_fcl(fcl* f);
int is_noisy(fcl* f);
void eliminate_noisy_layers(fcl* f);
void assign_noise_arrays(fcl* f, float** noise_biases, float** noise, int index);
void reinitialize_weights_according_to_scores_and_inner_info_fcl(fcl* f);
void memcopy_vector_to_indices(fcl* f, int* vector);
void memcopy_scores_to_indices(fcl* f, int* vector);
void free_scores(fcl* f);
void free_indices(fcl* f);
void assign_vector_to_scores(fcl* f, float* vector);
void set_null_scores(fcl* f);
void set_null_indices(fcl* f);
void memcopy_indices_to_vector(fcl* f, int* vector);
void reinitialize_weights_according_to_scores_fcl_only_percentage(fcl* f, float percentage);
void memcopy_vector_to_indices2(fcl* f, int* vector);
fcl* reset_fcl_only_for_ff(fcl* f);

#endif
