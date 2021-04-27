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

#ifndef __NORMALIZATION_H__
#define __NORMALIZATION_H__

void local_response_normalization_feed_forward(float* tensor,float* output, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k, int* used_kernels);
void local_response_normalization_back_prop(float* tensor,float* tensor_error,float* output_error, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k, int* used_kernels);
void batch_normalization_feed_forward(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon);
void batch_normalization_back_prop(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon);
void channel_normalization_feed_forward(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs,float epsilon, int rows_pad, int cols_pad, int rows, int cols, int* used_kernels);
void channel_normalization_back_prop(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs_error, float* gamma_error, float* beta_error, float* input_error, float** temp_vectors_error,float* temp_array, float epsilon, int rows_pad, int cols_pad, int rows, int cols, int* used_kernels);
void batch_normalization_final_mean_variance(float** input_vectors, int n_vectors, int vector_size, int mini_batch_size, bn* bn_layer);
void group_normalization_feed_forward(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, int pad_i, int pad_j, float* post_normalization, int* used_kernels);
void group_normalization_back_propagation(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, float* ret_error,int pad_i, int pad_j, float* input_error, int* used_kernels);
void batch_normalization_feed_forward_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon);
void batch_normalization_feed_forward_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon, int i);
void batch_normalization_back_prop_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon);
void batch_normalization_back_prop_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon, int j);
void normalize_scores_among_fcl_layers(fcl* f);
void normalize_scores_among_cl_layers(cl* f);
void normalize_scores_among_all_internal_layers(model* m);
void given_max_min_normalize_fcl(fcl* f, float max, float min);
void given_max_min_normalize_cl(cl* f, float max, float min);
void normalize_among_all_leyers(model* m);
void feed_forward_scaled_l2_norm(int input_dimension, float learned_g, float* norm, float* input, float* output);
void back_propagation_scaled_l2_norm(int input_dimension,float learned_g, float* d_learned_g, float norm,float* input, float* output_error, float* input_error);
void local_response_normalization_feed_forward_fcl(float* input,float* output,int size, float n_constant, float beta, float alpha, float k, int* used_outputs);
void local_response_normalization_back_prop_fcl(float* input,float* input_error,float* output_error, int size, float n_constant, float beta, float alpha, float k, int* used_kernels);
void group_normalization_feed_forward_without_learning_parameters(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, int pad_i, int pad_j, float* post_normalization, int* used_kernels, bn** bns2);
void group_normalization_back_propagation_without_learning_parameters(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, float* ret_error,int pad_i, int pad_j, float* input_error, int* used_kernels, bn** bns2);

#endif
