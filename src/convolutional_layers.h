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

#ifndef __CONVOLUTIONAL_LAYERS_H__
#define __CONVOLUTIONAL_LAYERS_H__

cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer);
int exists_d_kernels_cl(cl* c);
int exists_d_biases_cl(cl* c);
int exists_kernels_cl(cl* c);
int exists_biases_cl(cl* c);
int exists_pre_activation_cl(cl* c);
int exists_post_activation_cl(cl* c);
int exists_normalization_cl(cl* c);
int exists_pooling(cl* c);
int exists_edge_popup_stuff_cl(cl * c);
int exists_edge_popup_stuff_with_only_training_mode_cl(cl * c);
int exists_bp_handler_arrays(cl* c);
void free_convolutional(cl* c);
void save_cl(cl* f, int n);
void copy_cl_params(cl* f, float** kernels, float* biases);
cl* load_cl(FILE* fr);
cl* copy_cl(cl* f);
cl* reset_cl(cl* f);
cl* reset_cl_except_partial_derivatives(cl* f);
cl* reset_cl_without_dwdb(cl* f);
cl* reset_cl_for_edge_popup(cl* f);
uint64_t size_of_cls(cl* f);
void paste_cl(cl* f, cl* copy);
void paste_w_cl(cl* f, cl* copy);
void slow_paste_cl(cl* f, cl* copy,float tau);
uint64_t get_array_size_params_cl(cl* f);
uint64_t get_array_size_weights_cl(cl* f);
uint64_t get_array_size_scores_cl(cl* f);
void memcopy_vector_to_params_cl(cl* f, float* vector);
void memcopy_params_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_weights_cl(cl* f, float* vector);
void memcopy_weights_to_vector_cl(cl* f, float* vector);
void memcopy_scores_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_scores_cl(cl* f, float* vector);
void memcopy_vector_to_derivative_params_cl(cl* f, float* vector);
void memcopy_derivative_params_to_vector_cl(cl* f, float* vector);
void set_convolutional_biases_to_zero(cl* c);
void set_convolutional_unused_weights_to_zero(cl* c);
void sum_score_cl(cl* input1, cl* input2, cl* output);
void compare_score_cl(cl* input1, cl* input2, cl* output);
void compare_score_cl_with_vector(cl* input1, float* input2, cl* output);
void dividing_score_cl(cl* c,float value);
void reset_score_cl(cl* f);
void reinitialize_weights_according_to_scores_cl(cl* f, float percentage, float goodness);
void reinitialize_w_cl(cl* f);
cl* reset_edge_popup_d_cl(cl* f);
void set_low_score_cl(cl* f);
cl* convolutional_without_learning_parameters(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer);
void free_convolutional_without_learning_parameters(cl* c);
cl* copy_cl_without_learning_parameters(cl* f);
cl* reset_cl_without_learning_parameters(cl* f);
cl* reset_cl_without_dwdb_without_learning_parameters(cl* f);
uint64_t size_of_cls_without_learning_parameters(cl* f);
void paste_cl_without_learning_parameters(cl* f, cl* copy);
uint64_t count_weights_cl(cl* c);


#endif
