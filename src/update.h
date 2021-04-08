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
#ifndef __UPDATE_H__
#define __UPDATE_H__

void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam);
void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size);
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_residual_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_convolutional_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_fully_connected_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_batch_normalized_layer_nesterov(bn** bns,int n_bn, float lr, float momentum, int mini_batch_size);
void update_batch_normalized_layer_adam(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_batch_normalized_layer_adam_diff_grad(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam);
void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size);
void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2,float beta1_adam,float beta2_adam);
void update_lstm_layer_adam_diff_grad(rmodel* m,float lr,int mini_batch_size,float b1, float b2,float beta1_adam,float beta2_adam);
void update_residual_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long t,float beta1_adam,float beta2_adam);
void update_residual_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_convolutional_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam);
void update_convolutional_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_fully_connected_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam);
void update_fully_connected_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_batch_normalized_layer_radam(bn** bns, int n_bn, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam);
void update_batch_normalized_layer_adamod(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_lstm_layer_radam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam);
void update_lstm_layer_adamod(rmodel* m,float lr,int mini_batch_size,float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_scaled_l2_norm_nesterov(scaled_l2_norm* l, float lr, float momentum, int mini_batch_size);
void update_scaled_l2_norm_adam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam);
void update_scaled_l2_norm_adamod(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod);
void update_scaled_l2_norm_adam_diff_grad(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam);
void update_scaled_l2_norm_radam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam);
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void update_transformer(transformer* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* time);
void update_transformer_decoder(transformer_decoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* time);
void update_transformer_encoder(transformer_encoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* time);
void update_vae_model(vaemodel* vm, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void update_training_parameters(float* beta1, float* beta2, long long unsigned int* time_step, float start_beta1, float start_beta2);

#endif
