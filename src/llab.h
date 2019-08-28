/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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

#ifndef __LLAB_H__
#define __LLAB_H__

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>


#define N_NORMALIZATION 5
#define BETA_NORMALIZATION 0.75
#define ALPHA_NORMALIZATION 0.0001
#define K_NORMALIZATION 2
#define NESTEROV 1
#define ADAM 2
#define RADAM 3
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
#define GROUP_NORMALIZATION 3
#define BETA1_ADAM 0.9
#define BETA2_ADAM 0.999
#define EPSILON_ADAM 0.00000001
#define EPSILON 0.00000001
#define RADAM_THRESHOLD 4
#define NO_REGULARIZATION 0
#define L2_REGULARIZATION 1
#define NO_CONVOLUTION 1
#define CONVOLUTION 2
#define BATCH_NORMALIZATION_TRAINING_MODE 1
#define BATCH_NORMALIZATION_FINAL_MODE 2
#define STATEFUL 1
#define STATELESS 2
#define LEAKY_RELU_THRESHOLD 0.1
#define LSTM_RESIDUAL  1
#define LSTM_NO_RESIDUAL 0

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
    int group_norm_channels;
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
    bn** group_norm;//n_kernels/group_norm_channels
} cl;

typedef struct rl { //residual-layers
    int channels, input_rows, input_cols, n_cl;
    float* input;
    cl* cl_output;
    cl** cls;
} rl;

typedef struct lstm { //long short term memory layers
    int size,layer,dropout_flag_up, dropout_flag_right, window, residual_flag, norm_flag, n_grouped_cell;//dropout flag = 1 if dropout must be applied
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
    float** out_up;//window x size
    float dropout_threshold_up;
    float dropout_threshold_right;
    bn** bns;
    
    
    
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

typedef struct vaemodel{
    int latent_size;
    float* z;
    float* input;
    float* dmean;
    float* dstd;
    model* encoder;
    model* decoder;
} vaemodel;

typedef struct ganmodel{
	model* generator;
	model* discriminator;
	model* discriminator2;
	int mini_batch_size;
	int generator_gradient_descent_flag;
	int discriminator_gradient_descent_flag;
	int generator_regularization;
	int discriminator_regularization;
	int generator_total_number_weights;
	int discriminator_total_number_weights;
	float generator_lr;
	float discriminator_lr;
	float generator_momentum;
	float discriminator_momentum;
	float generator_b1;
	float discriminator_b1;
	float generator_b2;
	float discriminator_b2;
	float generator_lambda;
	float discriminator_lambda;
	unsigned long long int generator_t;
	unsigned long long int discriminator_t;
} ganmodel;

typedef struct thread_args_model {
    model* m;
    int rows,cols,channels,error_dimension;
    float* input;
    float* error;
    float** returning_error;
} thread_args_model;

typedef struct thread_args_rmodel {
    rmodel* m;
    float** hidden_states;
    float** cell_states;
    float** input_model;
    float** error_model;
    float**** returning_error;
    float*** ret_input_error;
} thread_args_rmodel;

typedef struct thread_args_vae_model {
    vaemodel* vm;
    int rows,cols,channels,error_dimension;
    float* input;
    float* error;
    float** returning_error;
} thread_args_vae_model;

typedef struct thread_args_gan_model {
    ganmodel* gm;
    int g_d_in,g_i_in,g_j_in,d_d_in,d_i_in,d_j_in,output_size;
    float* real_input;
    float* noise_input;
    float** ret_err;
} thread_args_gan_model;

// Functions defined in math.c

#include "batch_norm_layers.h"
#include "bmodel.h"
#include "clipping_gradient.h"
#include "convolutional.h"
#include "convolutional_layers.h"
#include "fully_connected.h"
#include "fully_connected_layers.h"
#include "gan_model.h"
#include "gd.h"
#include "math_functions.h"
#include "model.h"
#include "multi_core_gan_model.h"
#include "multi_core_model.h"
#include "multi_core_rmodel.h"
#include "multi_core_vae_model.h"
#include "normalization.h"
#include "recurrent.h"
#include "recurrent_layers.h"
#include "residual_layers.h"
#include "rmodel.h"
#include "utils.h"
#include "vae_model.h"

#endif
