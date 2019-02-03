#ifndef __LLAB_DT_H__
#define __LLAB_DT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONDITION_A(x,y) (x > y)
#define CONDITION_B(x,y) (x >= y)
#define CONDITION_C(x,y) (x == y)
#define CONDITION_D(x,y) (x <= y)
#define CONDITION_E(x,y) (x < y)
#define CONDITION_F(x,y) (!strcmp(x,y))


typedef struct decision_tree {
    int number_instances;// Number of total instances
    int char_feature_number, char_labels_number; // feature_number = 0 no char features, labels_number = 0 no char labels
    int int_feature_number, int_labels_number; // feature_number = 0 no int features, labels_number = 0 no int labels
    int float_feature_number, float_labels_number; // feature_number = 0 no char features, labels_number = 0 no char labels
    int char_condition_flag;//if the son is created with a char condition on char features (indicates on which char feature it's the condition)
    int int_condition_flag;//if the son is created with a float condition on float features (indicates on which char feature it's the condition)
    int float_condition_flag;//if the son is created with an int condition on int features (indicates on which int feature it's the condition)
    int unwanted_char_size;
    int unwanted_float_size;
    int unwanted_int_size;
    int char_second_dimension_max_size;
    char** char_features;//(number_instances*different_features)*char_second_dimension_max_size
    int* int_features;//number_instances*different_features
    float* float_labels;//number_instances*different_features
    char** char_labels;//(number_instances*different_labels)*char_second_dimension_max_size
    int* int_labels;//(number_instances*different_labels)
    float* float_labels;//(number_instances*different_labels)
    float impurity;
    float conditional_threshold;
    char* conditional_string;
    char** unwanted_conditional_list//unwanted_char_size*char_second_dimension_max_size
    int* unwanted_conditional_list//unwanted_int_size
    float* unwanted_conditional_list//unwanted_float_size
    decision_tree** sons;
} decision_tree;


// Functions defined in utils.c
float r2();
float drand ();
float random_normal ();
float random_general_gaussian(float mean, float n);
float random_general_gaussian_xavier_init(float mean, float n);
void get_dropout_array(int size, float* mask, float* input, float* output); //can be transposed in opencl
void set_dropout_mask(int size, float* mask, float threshold); //can be transposed in opencl
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
void read_file_in_char_vector(char** ksource, char* fname, int* size);
void dot1D(float* input1, float* input2, float* output, int size); //can be transposed in opencl
void copy_array(float* input, float* output, int size);//can be transposed in opencl
void sum1D(float* input1, float* input2, float* output, int size);//can be transposed in opencl
void mul_value(float* input, float value, float* output, int dimension);//can be transposed in opencl
void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void update_residual_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void update_convolutional_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void update_fully_connected_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);//can be transposed in opencl
void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3);//can be transoposed in opencl
void sum_residual_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);//can be transoposed in opencl
void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3);//can be transoposed in opencl
void sum_convolutional_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);//can be transoposed in opencl
void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3);//can be transoposed in opencl
void sum_fully_connected_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);//can be transoposed in opencl
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void update_residual_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void update_convolutional_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void update_fully_connected_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);//can be transoposed in opencl
void add_l2_residual_layer(model* m,int total_number_weights,float lambda);//can be transoposed in opencl
void add_l2_residual_layer_bmodel(bmodel* m,int total_number_weights,float lambda);//can be transoposed in opencl
void add_l2_convolutional_layer(model* m,int total_number_weights,float lambda);//can be transoposed in opencl
void add_l2_convolutional_layer_bmodel(bmodel* m,int total_number_weights,float lambda);//can be transoposed in opencl
void add_l2_fully_connected_layer(model* m,int total_number_weights,float lambda);//can be transoposed in opencl
void add_l2_fully_connected_layer_bmodel(bmodel* m,int total_number_weights,float lambda);//can be transoposed in opencl
int shuffle_char_matrices_float_int_vectors(char** m,char** m1,float* f, int* v,int n);
void copy_char_array(char* input, char* output, int size);
int shuffle_char_matrices_float_int_int_vectors(char** m,char** m1,float* f, int* v, int* v2, int n);
void update_batch_normalized_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size);
void update_batch_normalized_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2);

#endif __LLAB_DT_H__
