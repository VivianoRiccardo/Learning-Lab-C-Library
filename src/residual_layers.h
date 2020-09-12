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

#ifndef __RESIDUAL_LAYERS_H__
#define __RESIDUAL_LAYERS_H__

rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls);
void free_residual(rl* r);
void save_rl(rl* f, int n);
void heavy_save_rl(rl* f, int n);
rl* load_rl(FILE* fr);
rl* heavy_load_rl(FILE* fr);
rl* copy_rl(rl* f);
void paste_rl(rl* f, rl* copy);
rl* reset_rl(rl* f);
unsigned long long int size_of_rls(rl* f);
void slow_paste_rl(rl* f, rl* copy,float tau);
int get_array_size_params_rl(rl* f);
void memcopy_vector_to_params_rl(rl* f, float* vector);
void memcopy_params_to_vector_rl(rl* f, float* vector);
void memcopy_vector_to_derivative_params_rl(rl* f, float* vector);
void memcopy_derivative_params_to_vector_rl(rl* f, float* vector);
void set_residual_biases_to_zero(rl* r);
int rl_adjusting_weights_after_edge_popup(rl* c, int* used_input, int* used_output);
int* get_used_kernels_rl(rl* c, int* used_input);
int* get_used_channels_rl(rl* c, int* used_output);
void paste_w_rl(rl* f, rl* copy);
void sum_score_rl(rl* input1, rl* input2, rl* output);
void dividing_score_rl(rl* f, float value);
void reset_score_rl(rl* f);
void reinitialize_scores_rl(rl* f, float percentage, float goodness);
void free_residual_for_edge_popup(rl* r);
rl* light_load_rl(FILE* fr);
rl* light_reset_rl(rl* f);
int get_array_size_weights_rl(rl* f);
void memcopy_vector_to_scores_rl(rl* f, float* vector);
void memcopy_scores_to_vector_rl(rl* f, float* vector);
rl* copy_light_rl(rl* f);
rl* reset_rl_for_edge_popup(rl* f);
rl* reset_rl_without_dwdb(rl* f);
void paste_rl_for_edge_popup(rl* f, rl* copy);
void free_residual_complementary_edge_popup(rl* r);
void memcopy_weights_to_vector_rl(rl* f, float* vector);
void memcopy_vector_to_weights_rl(rl* f, float* vector);
void compare_score_rl(rl* input1, rl* input2, rl* output);

#endif
