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

#ifndef __RECURRENT_LAYERS_H__
#define __RECURRENT_LAYERS_H__

lstm* recurrent_lstm(int input_size, int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag);
void free_recurrent_lstm(lstm* rlstm);
void save_lstm(lstm* rlstm, int n);
lstm* load_lstm(FILE* fr);
lstm* copy_lstm(lstm* l);
void paste_lstm(lstm* l,lstm* copy);
void slow_paste_lstm(lstm* l,lstm* copy, float tau);
lstm* reset_lstm(lstm* f);
uint64_t get_array_size_params_lstm(lstm* f);
void memcopy_vector_to_params_lstm(lstm* f, float* vector);
void memcopy_params_to_vector_lstm(lstm* f, float* vector);
void memcopy_vector_to_derivative_params_lstm(lstm* f, float* vector);
void memcopy_derivative_params_to_vector_lstm(lstm* f, float* vector);
void paste_w_lstm(lstm* l,lstm* copy);
void heavy_save_lstm(lstm* rlstm, int n);
lstm* heavy_load_lstm(FILE* fr);
void get_used_outputs_lstm(int* arr, int input, int output, int* indices, float k_percentage);
lstm* recurrent_lstm_without_learning_parameters (int input_size,int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag);
void free_recurrent_lstm_without_learning_parameters(lstm* rlstm);
lstm* copy_lstm_without_learning_parameters(lstm* l);
lstm* reset_lstm_without_learning_parameters(lstm* f);
lstm* reset_lstm_except_partial_derivatives(lstm* f);
lstm* reset_lstm_without_dwdb(lstm* f);
lstm* reset_lstm_without_dwdb_without_learning_parameters(lstm* f);
uint64_t size_of_lstm(lstm* l);
uint64_t size_of_lstm_without_learning_parameters(lstm* l);
void paste_lstm_without_learning_parameters(lstm* l,lstm* copy);
uint64_t count_weights_lstm(lstm* l);
uint64_t get_array_size_params_lstm(lstm* f);
uint64_t get_array_size_scores_lstm(lstm* f);
uint64_t get_array_size_weights_lstm(lstm* f);
void memcopy_params_to_vector_lstm(lstm* f, float* vector);
void memcopy_scores_to_vector_lstm(lstm* f, float* vector);
void memcopy_vector_to_params_lstm(lstm* f, float* vector);
void memcopy_vector_to_weights_lstm(lstm* f, float* vector);
void memcopy_weights_to_vector_lstm(lstm* f, float* vector);
void memcopy_vector_to_scores_lstm(lstm* f, float* vector);

#endif
