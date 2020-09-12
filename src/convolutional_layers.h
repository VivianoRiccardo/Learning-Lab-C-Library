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

cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int layer);
void free_convolutional(cl* c);
void save_cl(cl* f, int n);
void copy_cl_params(cl* f, float** kernels, float* biases);
cl* load_cl(FILE* fr);
cl* copy_cl(cl* f);
void paste_cl(cl* f, cl* copy);
cl* reset_cl(cl* f);
unsigned long long int size_of_cls(cl* f);
void slow_paste_cl(cl* f, cl* copy,float tau);
int get_array_size_params_cl(cl* f);
void memcopy_params_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_params_cl(cl* f, float* vector);
void memcopy_derivative_params_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_derivative_params_cl(cl* f, float* vector);
void set_convolutional_biases_to_zero(cl* c);
void set_convolutional_unused_weights_to_zero(cl* c);
int cl_adjusting_weights_after_edge_popup(cl* c, int* used_input, int* used_output);
int* get_used_channels(cl* c, int* ch);
void paste_w_cl(cl* f, cl* copy);
void heavy_save_cl(cl* f, int n);
cl* heavy_load_cl(FILE* fr);
void sum_score_cl(cl* input1, cl* input2, cl* output);
void dividing_score_cl(cl* c,float value);
void reset_score_cl(cl* f);
void reinitialize_scores_cl(cl* f, float percentage, float goodness);
void free_convolutional_for_edge_popup(cl* c);
cl* light_load_cl(FILE* fr);
cl* light_reset_cl(cl* f);
int get_array_size_weights_cl(cl* f);
void memcopy_scores_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_scores_cl(cl* f, float* vector);
cl* copy_light_cl(cl* f);
cl* reset_cl_for_edge_popup(cl* f);
cl* reset_cl_without_dwdb(cl* f);
void paste_cl_for_edge_popup(cl* f, cl* copy);
void free_convolutional_complementary_edge_popup(cl* c);
void memcopy_weights_to_vector_cl(cl* f, float* vector);
void memcopy_vector_to_weights_cl(cl* f, float* vector);
void compare_score_cl(cl* input1, cl* input2, cl* output);

#endif
