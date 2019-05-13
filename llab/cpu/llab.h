#ifndef __LLAB_H__
#define __LLAB_H__

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <string.h>
#include <time.h>


#define N_NORMALIZATION 5
#define BETA_NORMALIZATION 0.75
#define ALPHA_NORMALIZATION 0.0001
#define K_NORMALIZATION 2
#define NESTEROV 1
#define ADAM 2
#define FCLS 1
#define CLS 2
#define RLS 3
#define BNS 4
#define LSTMS 1
#define NO_ACTIVATION 0
#define SIGMOID 1
#define RELU 2
#define SOFTMAX 3
#define TANH 4
#define LEAKY_RELU 5
#define NO_POOLING 0
#define MAX_POOLING 1
#define AVARAGE_POOLING 2
#define NO_DROPOUT 0
#define DROPOUT 1
#define DROPOUT_TEST 2
#define NO_NORMALIZATION 0
#define LOCAL_RESPONSE_NORMALIZATION 1
#define BATCH_NORMALIZATION 2
#define BETA1_ADAM 0.9
#define BETA2_ADAM 0.999
#define EPSILON_ADAM 0.00000001
#define EPSILON 0.00000001
#define NO_REGULARIZATION 0
#define L2_REGULARIZATION 1
#define NO_CONVOLUTION 1
#define CONVOLUTION 2
#define BATCH_NORMALIZATION_TRAINING_MODE 1
#define BATCH_NORMALIZATION_FINAL_MODE 2
#define STATEFUL 1
#define STATELESS 2

/* LAYERS MUST START FROM 0*/
typedef struct fcl { //fully-connected-layers
    int input,output,layer,dropout_flag;//dropout flag = 1 if dropout must be applied
    int activation_flag; // activation flag = 0 -> no activation, flag = 1 -> sigmoid, = 2 -> relu, = 3 -> softmax, 4->tanhh
    float* weights;// output*input
    float* d_weights;// output*input
    float* d1_weights;// output*input
    float* d2_weights;// output*input
    float* biases; //output
    float* d_biases; //output
    float* d1_biases; //output
    float* d2_biases; //output
    float* pre_activation; //output
    float* post_activation; //output
    float* dropout_mask;//output
    float* dropout_temp;//output
    float* temp;//output
    float* temp3;//output
    float* temp2;//input
    float* error2;//input
    float dropout_threshold;
} fcl;

/* PADDING_ROWS MUST BE = PADDING_COLS AND ALSO STRIDE_ROWS = STRIDE_COLS*/
typedef struct cl { //convolutional-layers
    int channels, input_rows, input_cols,layer, convolutional_flag;
    int kernel_rows, kernel_cols, n_kernels;
    int stride1_rows, stride1_cols, padding1_rows, padding1_cols;
    int stride2_rows, stride2_cols, padding2_rows, padding2_cols;
    int pooling_rows, pooling_cols;
    int normalization_flag, activation_flag, pooling_flag; // activation flag = 0, no activation, = 1 sigmoid, = 2 relu, pooling flag = 1 max-pooling, = 2 avarage-pooling
    int rows1, cols1, rows2,cols2;
    float** kernels; //n_kernels - channels*kernel_rows*kernel_cols
    float** d_kernels; //n_kernels - channels*kernel_rows*kernel_cols
    float** d1_kernels; //n_kernels - channels*kernel_rows*kernel_cols
    float** d2_kernels; //n_kernels - channels*kernel_rows*kernel_cols
    float* biases; //n_kernels
    float* d_biases; //n_kernels
    float* d1_biases; //n_kernels
    float* d2_biases; //n_kernels
    float* pre_activation;//n_kernels*((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows)*((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols)
    float* post_activation;//n_kernels*((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows)*((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols)
    float* post_normalization;//n_kernels*((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows)*((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols)
    float* post_pooling;//n_kernels*((((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows) - pooling_rows)/stride2_rows + 1 + 2*padding2_rows)*((((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols) - pooling_cols)/stride2_cols + 1 + 2*padding2_cols)
    float* temp;//n_kernels*rows1*cols1
    float* temp2;//n_kernels*rows1*cols1
    float* temp3;//n_kernels*rows1*cols1
    float* error2;//channels*input_rows*input_cols
} cl;

typedef struct rl { //residual-layers
    int channels, input_rows, input_cols, n_cl;
    float* input;
    cl* cl_output;
    cl** cls;
} rl;


typedef struct bn{//batch_normalization layer
    int batch_size, vector_dim, layer, activation_flag, mode_flag;
    float epsilon;
    float** input_vectors;//batch_size*vector_dim
    float** temp_vectors;//batch_size*vector_dim
    float* gamma;//vector_dim
    float* d_gamma;//vector_dim
    float* d1_gamma;//vector_dim
    float* d2_gamma;//vector_dim
    float* beta;//vector_dim
    float* d_beta;//vector_dim
    float* d1_beta;//vector_dim
    float* d2_beta;//vector_dim
    float* mean;//vector_dim
    float* var;//vector_dim
    float** outputs;//batch_size*vector_dim
    float** error2;//batch_size*vector_dim
    float** temp1;//batch_size*vector_dim
    float* temp2;//vector_dim
    float** post_activation;//batch_size*vector_dim
    float* final_mean;//vector_dim
    float* final_var;//vector_dim
}bn;

typedef struct lstm { //long short term memory layers
    int size,layer,dropout_flag_up, dropout_flag_right, window;//dropout flag = 1 if dropout must be applied
    float** w;// 4 x size*size
    float** u;// 4 x size*size
    float** d_w;// 4 x size*size
    float** d1_w;// 4 x size*size
    float** d2_w;// 4 x size*size
    float** d_u;// 4 x size*size
    float** d1_u;// 4 x size*size
    float** d2_u;// 4 x size*size
    float** biases; //4 x size
    float** d_biases; //4 x size
    float** d1_biases; //4 x size
    float** d2_biases; //4 x size
    float*** lstm_z; //window x 4 x size
    float** lstm_hidden; //window x size
    float** lstm_cell; //window x size
    float* dropout_mask_up;//size
    float* dropout_mask_right;//size
    float dropout_threshold_up;
    float dropout_threshold_right;
} lstm;

typedef struct model {
    int layers, n_rl, n_cl, n_fcl;
    rl** rls;//rls = residual-layers
    cl** cls;//cls = convolutional-layers
    fcl** fcls; // fcls = fully-connected-layers
    int** sla; //layers*layers, 1 for fcls, 2 for cls, 3 for rls, sla = sequential layers array
} model;

typedef struct bmodel {
    int layers, n_rl, n_cl, n_fcl, n_bn;
    rl** rls;//rls = residual-layers
    cl** cls;//cls = convolutional-layers
    fcl** fcls; // fcls = fully-connected-layers
    bn** bns; // bn = batch-normalization layer
    int** sla; //layers*layers, 1 for fcls, 2 for cls, 3 for rls, 4 = batch normalization sla = sequential layers array
} bmodel;

typedef struct rmodel {
    int layers, n_lstm, window, hidden_state_mode;
    lstm** lstms;
    int** sla;
} rmodel;

// Functions defined in math.c
void softmax(float* input, float* output, int size);
float sigmoid(float x);
void sigmoid_array(float* input, float* output, int size);//can be transposed in opencl
float derivative_sigmoid(float x);
void derivative_sigmoid_array(float* input, float* output, int size);//can be transposed in opencl
float relu(float x);
void relu_array(float* input, float* output, int size);//can be transposed in opencl
float derivative_relu(float x);
void derivative_relu_array(float* input, float* output, int size);//can be transposed in opencl
float leaky_relu(float x);
void leaky_relu_array(float* input, float* output, int size);//can be transposed in opencl
float derivative_leaky_relu(float x);
void derivative_leaky_relu_array(float* input, float* output, int size);//can be transposed in opencl
float tanhh(float x);
void tanhh_array(float* input, float* output, int size);//can be transposed in opencl
float derivative_tanhh(float x);
void derivative_tanhh_array(float* input, float* output, int size);//can be transposed in opencl
float mse(float y_hat, float y);
float derivative_mse(float y_hat, float y);
float cross_entropy(float y_hat, float y);
float derivative_cross_entropy(float y_hat, float y);
float cross_entropy_reduced_form(float y_hat, float y);
float derivative_cross_entropy_reduced_form_with_softmax(float y_hat, float y);
void derivative_cross_entropy_reduced_form_with_softmax_array(float* y_hat, float* y,float* output, int size);//can be transposed in opencl

// Functions defined in fully_connected.c
void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size);//can be transposed in opencl
void fully_connected_back_prop(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size);//can be transposed in opencl


// Functions defined in convolutional.c
void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride, int padding);//can be transposed in opencl
void convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride, int padding);//can be transposed in opencl
void max_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride, int padding);//can be transposed in opencl
void max_pooling_back_prop(float* input, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride, int padding, float* input_error);//can be transposed in opencl
void avarage_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride, int padding);//can be transposed in opencl
void avarage_pooling_back_prop(float* input_error, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride, int padding);//can be transposed in opencl


// Functions defined in normalization.c
void local_response_normalization_feed_forward(float* tensor,float* output, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k);//can be transposed in opencl
void local_response_normalization_back_prop(float* tensor,float* tensor_error,float* output_error, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k);//can be transposed in opencl
void batch_normalization_feed_forward(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon);
void batch_normalization_back_prop(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon);
void batch_normalization_final_mean_variance(float** input_vectors, int n_vectors, int vector_size, int mini_batch_size, bn* bn_layer);

// Functions defined in gd.c
void nesterov_momentum(float* p, float lr, float m, int mini_batch_size, float dp, float* delta);
void adam_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size);


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
void free_matrix(float** m, int n);
void add_l2_lstm_layer(rmodel* m,int total_number_weights,float lambda);
void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size);
void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2);

// Functions defined in layers.c
fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold);
void free_fully_connected(fcl* f);
cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int layer, int convolutional_flag);
void free_convolutional(cl* c);
rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls);
void free_residual(rl* r);
void save_fcl(fcl* f, int n);
void copy_fcl_params(fcl* f, float* weights, float* biases);
fcl* load_fcl(FILE* fr);
void save_cl(cl* f, int n);
void copy_cl_params(cl* f, float** kernels, float* biases);
cl* load_cl(FILE* fr);
void save_rl(rl* f, int n);
rl* load_rl(FILE* fr);
fcl* copy_fcl(fcl* f);
cl* copy_cl(cl* f);
rl* copy_rl(rl* f);
void paste_fcl(fcl* f, fcl* copy);
void paste_cl(cl* f, cl* copy);
void paste_rl(rl* f, rl* copy);
fcl* reset_fcl(fcl* f);
cl* reset_cl(cl* f);
rl* reset_rl(rl* f);
unsigned long long int size_of_fcls(fcl* f);
unsigned long long int size_of_cls(cl* f);
unsigned long long int size_of_rls(rl* f);
bn* batch_normalization(int batch_size, int vector_input_dimension, int layer, int activation_flag);
void free_batch_normalization(bn* b);
void save_bn(bn* b, int n);
bn* load_bn(FILE* fr);
bn* copy_bn(bn* b);
bn* reset_bn(bn* b);
unsigned long long int size_of_bn(bn* b);
void paste_bn(bn* b1, bn* b2);
void slow_paste_fcl(fcl* f,fcl* copy, float tau);
void slow_paste_cl(cl* f, cl* copy,float tau);
void slow_paste_rl(rl* f, rl* copy,float tau);
void slow_paste_bn(bn* f, bn* copy,float tau);

// Functions defined in model.c
model* network(int layers, int n_rl, int n_cl, int n_fcl, rl** rls, cl** cls, fcl** fcls);
void free_model(model* m);
model* copy_model(model* m);
void save_model(model* m, int n);
model* load_model(char* file);
void ff_fcl_fcl(fcl* f1, fcl* f2);
void ff_fcl_cl(fcl* f1, cl* f2);
void ff_cl_fcl(cl* f1, fcl* f2);
void ff_cl_cl(cl* f1, cl* f2);
float* bp_fcl_fcl(fcl* f1, fcl* f2, float* error);
float* bp_fcl_cl(fcl* f1, cl* f2, float* error);
float* bp_cl_cl(cl* f1, cl* f2, float* error);
float* bp_cl_fcl(cl* f1, fcl* f2, float* error);
void model_tensor_input_ff(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input);
float* model_tensor_input_bp(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension);
model* reset_model(model* m);
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda);
void sum_model_partial_derivatives(model* m, model* m2, model* m3);
unsigned long long int size_of_model(model* m);
void paste_model(model* m, model* copy);
int count_weights(model* m);
void slow_paste_model(model* m, model* copy, float tau);

// Functions defined in clipping_gradient.c
void clipping_gradient(model* m, float threshold);
void clip_rls(rl** rls, int n, float threshold,float norm);
void clip_cls(cl** cls, int n, float threshold, float norm);
void clip_fcls(fcl** fcls, int n, float threshold, float norm);
float sum_all_quadratic_derivative_weights_rls(rl** rls, int n);
float sum_all_quadratic_derivative_weights_cls(cl** cls, int n);
float sum_all_quadratic_derivative_weights_fcls(fcl** fcls, int n);

// Functions defined in bmodel.c
bmodel* batch_network(int layers, int n_rl, int n_cl, int n_fcl, int n_bnl, rl** rls, cl** cls, fcl** fcls, bn** bnls);
void free_bmodel(bmodel* m);
bmodel* copy_bmodel(bmodel* m);
void paste_bmodel(bmodel* m, bmodel* copy);
void slow_paste_bmodel(bmodel* m, bmodel* copy, float tau);
bmodel* reset_bmodel(bmodel* m);
unsigned long long int size_of_bmodel(bmodel* m);
void save_bmodel(bmodel* m, int n);
bmodel* load_bmodel(char* file);
int count_bmodel_weights(bmodel* m);
void update_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda);
void sum_model_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);

// Functions defined in recurrent.c
void lstm_ff(float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int size);
float** lstm_bp(int flag, int size, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus);


// Functions defined in recurrent_layers.c
lstm* recurrent_lstm(int size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window);
void free_recurrent_lstm(lstm* rlstm);
void save_lstm(lstm* rlstm, int n);
lstm* load_lstm(FILE* fr);
lstm* copy_lstm(lstm* l);
void paste_lstm(lstm* l,lstm* copy);
void slow_paste_lstm(lstm* l,lstm* copy, float tau);
lstm* reset_lstm(lstm* f);

// Functions defined in rmodel.c
rmodel* recurrent_network(int layers, int n_lstm, lstm** lstms, int window, int hidden_state_mode);
void free_rmodel(rmodel* m);
rmodel* copy_rmodel(rmodel* m);
void paste_rmodel(rmodel* m, rmodel* copy);
void slow_paste_rmodel(rmodel* m, rmodel* copy, float tau);
rmodel* reset_rmodel(rmodel* m);
void save_rmodel(rmodel* m, int n);
rmodel* load_rmodel(char* file);
void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, rmodel* m);
float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m);
int count_weights_rmodel(rmodel* m);
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda);
#endif
