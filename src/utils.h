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

#ifndef __UTILS_H__
#define __UTILS_H__

char* get_full_path(char* directory, char* filename);
float r2();
float drand ();
float random_normal ();
float random_general_gaussian(float mean, float n);
float random_general_gaussian_xavier_init(float mean, float n);
void get_dropout_array(int size, float* mask, float* input, float* output); 
void set_dropout_mask(int size, float* mask, float threshold); 
void ridge_regression(float *dw, float w, float lambda, int n);
int read_files(char** name, char* directory);
char* itoa(int i, char b[]);
int shuffle_char_matrix(char** m,int n);
int bool_is_real(float d);
int shuffle_float_matrix(float** m,int n);
int shuffle_int_matrix(int** m,int n);
int shuffle_char_matrices(char** m,char** m1,int n);
int shuffle_float_matrices(float** m,float** m1,int n);
int shuffle_int_matrices(int** m,int** m1,int n);
int read_file_in_char_vector(char** ksource, char* fname, int* size);
void dot1D(float* input1, float* input2, float* output, int size); 
void copy_array(float* input, float* output, int size);
void sum1D(float* input1, float* input2, float* output, int size);
void mul_value(float* input, float value, float* output, int dimension);
void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_residual_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);
void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_convolutional_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);
void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_fully_connected_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);
void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_residual_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);
void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_convolutional_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);
void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_fully_connected_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);
void update_residual_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);
void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);
void update_convolutional_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);
void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);
void update_fully_connected_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);
void add_l2_residual_layer(model* m,int total_number_weights,float lambda);
void add_l2_residual_layer_bmodel(bmodel* m,int total_number_weights,float lambda);
void add_l2_convolutional_layer(model* m,int total_number_weights,float lambda);
void add_l2_convolutional_layer_bmodel(bmodel* m,int total_number_weights,float lambda);
void add_l2_fully_connected_layer(model* m,int total_number_weights,float lambda);
void add_l2_fully_connected_layer_bmodel(bmodel* m,int total_number_weights,float lambda);
int shuffle_char_matrices_float_int_vectors(char** m,char** m1,float* f, int* v,int n);
void copy_char_array(char* input, char* output, int size);
int shuffle_char_matrices_float_int_int_vectors(char** m,char** m1,float* f, int* v, int* v2, int n);
void update_batch_normalized_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);
void update_batch_normalized_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);
void free_matrix(float** m, int n);
void add_l2_lstm_layer(rmodel* m,int total_number_weights,float lambda);
void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size);
void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2);
void sum_lstm_layers_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3);
void update_residual_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long t);
void update_residual_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t);
void update_convolutional_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t);
void update_convolutional_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t);
void update_fully_connected_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t);
void update_fully_connected_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2,unsigned long long int t);
void update_batch_normalized_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t);
void update_lstm_layer_radam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, unsigned long long int t);
long long unsigned int** confusion_matrix(float* model_output, float* real_output, long long unsigned int** cm, int size, float threshold);
double* accuracy_array(long long unsigned int** cm, int size);
int shuffle_float_matrices_float_int_int_vectors(float** m,float** m1,float* f, int* v, int* v2, int n);
int shuffle_float_matrices_float_int_vectors(float** m,float** m1,float* f, int* v,int n);
double* precision_array(long long unsigned int** cm, int size);
double* sensitivity_array(long long unsigned int** cm, int size);
double* specificity_array(long long unsigned int** cm, int size);
void print_accuracy(long long unsigned int** cm, int size);
void print_precision(long long unsigned int** cm, int size);
void print_sensitivity(long long unsigned int** cm, int size);
void print_specificity(long long unsigned int** cm, int size);

#endif
