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
void get_dropout_array(int size, float* mask, float* input, float* output); 
void set_dropout_mask(int size, float* mask, float threshold); 
void ridge_regression(float *dw, float w, float lambda_value, int n);
int read_files(char** name, char* directory);
char* itoa_n(int i, char b[]);
int shuffle_char_matrix(char** m,int n);
int bool_is_real(float d);
int shuffle_float_matrix(float** m,int n);
int shuffle_int_matrix(int** m,int n);
int shuffle_char_matrices(char** m,char** m1,int n);
int shuffle_float_matrices(float** m,float** m1,int n);
int shuffle_int_matrices(int** m,int** m1,int n);
int read_file_in_char_vector(char** ksource, char* fname, int* size);
void copy_array(float* input, float* output, int size);
int shuffle_char_matrices_float_int_vectors(char** m,char** m1,float* f, int* v,int n);
void copy_char_array(char* input, char* output, int size);
int shuffle_char_matrices_float_int_int_vectors(char** m,char** m1,float* f, int* v, int* v2, int n);
void free_matrix(void** m, int n);
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
void quick_sort(float A[], int I[], int lo, int hi);
void copy_int_array(int* input, int* output, int size);
int shuffle_int_array(int* m,int n);
char** get_files(int index1, int n_files);
int check_nans_matrix(float** m, int rows, int cols);
void merge(float* values, int* indices, int temp[], int from_index, int mid, int to, int length);
void merge_sort(float* values, int* indices, int low, int high);
void sort(float* values, int* indices, int low, int high);
void free_tensor(float*** t, int dim1, int dim2);
int shuffle_float_matrix_float_tensor(float** m,float*** t,int n);
void set_vector_with_value(float value, float* v, int dimension);
char* read_files_from_file(char* file, int package_size);
void set_files_free_from_file(char* file_to_free, char* file);
void remove_occupied_sets(char* file);
int msleep(long msec);
int* get_new_copy_int_array(int* array, int size);
void set_int_vector_with_value(int value, int* v, int dimension);
int argmax(float* vector, int dimension);
void max_heapify(float* values, uint* indices,uint* reverse_indices, uint n, uint i);
void max_heapify_up(float* values, uint* indices,uint* reverse_indices,  uint n, uint i);
void remove_ith_element_from_max_heap(float* values, uint* indices,uint* reverse_indices,  uint n, uint i);
int shuffle_float_matrices(float** m,float** m1,int n);
void update_recursive_cumulative_heap_up(float* values, uint index, uint started_index, uint n, float value);
uint weighted_random_sample(float* cumulative_values, float* current_values, uint index, uint size, float random_value, double sum, uint* taken_values, uint taken_values_length);
int index_is_inside_buffer(uint* buffer, uint length, uint index);
int value_is_child(uint child, uint parent);
float subtracted_value(uint index, float* current_values, uint* taken_values, uint taken_values_length);
uint weighted_random_sample_rewards(float* cumulative_values, int* current_values, uint index, uint size, float random_value, double sum, uint* taken_values, uint taken_values_length, float alpha);
float subtracted_value_rewards(uint index, int* current_values, uint* taken_values, uint taken_values_length,float alpha);
int is_little_endian();
void reverse_ptr(void* ptr, uint64_t size);
void swap_array_bytes_order(void* ptr, uint64_t size, uint64_t len);
void convert_data(void* ptr, uint64_t size, uint64_t len);
void convert_communication_data(void* ptr, uint64_t size, uint64_t len);
int shuffle_int_array_until_length(int* m,int n, int length);
void quick_sort_int(int A[], int I[], int lo, int hi);
uint64_t uint64t_read_file_in_char_vector(char** ksource, char* fname, uint64_t* size);
void set_array_random_normal(float* array, int size);
void merge_for_probabilities(float* p, int* index, int left, int middle, int right, int n);
void merge_sort_for_probabilities(float* p, int* index, int left, int right, int n);
void set_float_vector_to_int_vector(float* v1, int* v2, int size, float step);
void readapt_scores(int** indices, int* index, int size_1, int size_2, dueling_categorical_dqn* dqn);
void scores_and_indices_readapter(int** indices, float** scores, int size_1, int size_2, float step, dueling_categorical_dqn* dqn);
void readapt_scores_float(float** indices, int* index, int size_1, int size_2, dueling_categorical_dqn* dqn);
void scores_and_indices_readapter_float(int** indices, float** scores, int size_1, int size_2, float step, dueling_categorical_dqn* dqn);


#endif
