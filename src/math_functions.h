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

#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__


void softmax(float* input, float* output, int size);
void derivative_softmax_array(int* input, float* output,float* softmax_arr,float* error, int size);
float sigmoid(float x);
void sigmoid_array(float* input, float* output, int size);
float derivative_sigmoid(float x);
void derivative_sigmoid_array(float* input, float* output, int size);
float relu(float x);
void relu_array(float* input, float* output, int size);
float derivative_relu(float x);
void derivative_relu_array(float* input, float* output, int size);
float leaky_relu(float x);
void leaky_relu_array(float* input, float* output, int size);
float derivative_leaky_relu(float x);
void derivative_leaky_relu_array(float* input, float* output, int size);
float tanhh(float x);
void tanhh_array(float* input, float* output, int size);
float derivative_tanhh(float x);
void derivative_tanhh_array(float* input, float* output, int size);
float mse(float y_hat, float y);
float derivative_mse(float y_hat, float y);
float cross_entropy(float y_hat, float y);
float derivative_cross_entropy(float y_hat, float y);
float cross_entropy_reduced_form(float y_hat, float y);
float derivative_cross_entropy_reduced_form_with_softmax(float y_hat, float y);
void derivative_cross_entropy_reduced_form_with_softmax_array(float* y_hat, float* y,float* output, int size);
float huber_loss(float y_hat, float y, float threshold);
float derivative_huber_loss(float y_hat, float y, float threshold);
void derivative_huber_loss_array(float* y_hat, float* y,float* output, float threshold, int size);
float modified_huber_loss(float y_hat, float y, float threshold1, float threshold2);
float derivative_modified_huber_loss(float y_hat, float y, float threshold1, float threshold2);
void derivative_modified_huber_loss_array(float* y_hat, float* y, float threshold1, float* output, float threshold2, int size);
float focal_loss(float y_hat, float y, float gamma);
void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size);
float derivative_focal_loss(float y_hat, float y, float gamma);
void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size);
void mse_array(float* y_hat, float* y, float* output, int size);
void derivative_mse_array(float* y_hat, float* y, float* output, int size);
void cross_entropy_array(float* y_hat, float* y, float* output, int size);
void derivative_cross_entropy_array(float* y_hat, float* y, float* output, int size);
void kl_divergence(float* input1, float* input2, float* output, int size);
void derivative_kl_divergence(float* y_hat, float* y, float* output, int size);
float entropy(float y_hat);
void entropy_array(float* y_hat, float* output, int size);
float derivative_entropy(float y_hat);
void derivative_entropy_array(float* y_hat, float* output, int size);
float abs_sigmoid(float x);
void abs_sigmoid_array(float* input, float* output, int size);
void softmax_array_not_complete(float* input, float* output,int* mask, int size);
float elu(float z, float a);
void elu_array(float* input, float* output, int size, float a);
float derivative_elu(float z, float a);
void derivative_elu_array(float* input, float* output, int size, float a);
void derivative_softmax(float* output,float* softmax_arr,float* error, int size);
void dot1D(float* input1, float* input2, float* output, int size);
void sum1D(float* input1, float* input2, float* output, int size);
void mul_value(float* input, float value, float* output, int dimension);
void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3);
void sum_lstm_layers_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3);
float float_abs(float a);
void float_abs_array(float* a, int n);
float* get_float_abs_array(float* a, int n);
void dot_float_input(float* input1, int* input2, float* output, int size);
void sum_model_partial_derivatives(model* m, model* m2, model* m3);
void sum_models_partial_derivatives(model* sum_m, model** models, int n_models);
void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3);
void sum_rmodels_partial_derivatives(rmodel* m, rmodel** m2, int n_models);
void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3);
int min_int(int x, int y);
int max_int(int x, int y);
double sum_over_input(float* inputs, int dimension);
float derivative_sigmoid_given_the_sigmoid(float x);
void derivative_sigmoid_array_given_the_sigmoid(float* input, float* output, int size);
float total_variation_loss_2d(float* y, int rows, int cols);
void derivative_total_variation_loss_2d(float* y, float* output, int rows, int cols);
void div1D(float* input1, float* input2, float* output, int size);
void sub1D(float* input1, float* input2, float* output, int size);
void inverse(float* input, float* output, int size);
float min_float(float x, float y);
float max_float(float x, float y);
float constrantive_loss(float y_hat, float y, float margin);
float derivative_constrantive_loss(float y_hat, float y, float margin);
void constrantive_loss_array(float* y_hat, float* y,float* output, float margin, int size);
void derivative_constrantive_loss_array(float* y_hat, float* y,float* output, float margin, int size);
float dotProduct1D(float* input1, float* input2, int size);
void additional_mul_value(float* input, float value, float* output, int dimension);
void copy_clipped_vector(float* vector, float* output, float maximum, float minimum, int dimension);
void clip_vector(float* vector, float minimum, float maximum, int dimension);
float mean(float* v, int size);
void sum_dueling_categorical_dqn_partial_derivatives(dueling_categorical_dqn* m1, dueling_categorical_dqn* m2, dueling_categorical_dqn* m3);
float factorised_gaussian();
void set_factorised_noise(int input, int output, float* noise, float* biases_noise);
float std(float* v, float mean, int size);
double normal_cdf(double x);
double calc_prob(double xi, double si, double max_xi);
int* get_sorted_probability_vector(float* means, float* std, int size, int* index);
void derivative_inverse_q_function_array(float* current_q, float* next_q, float* output,float* action, float alpha1, float alpha2, float gamma, int size);
int sample_softmax_with_temperature(float* input,float temperature, int size);
void add_value(float* input, float value, float* output, int dimension);
void compute_prob_average(float* input, float* output, int size);
void mat_mul(float* mat1, float* mat2,float* mat3, int size1, int size2, int size3);
void compute_prob_average_double(double* input, float* output, int size);
double sum_over_input_double(double* inputs, int dimension);
float compute_kl_qr_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip);
void huber_loss_array(float* y_hat, float* y,float* output, float threshold, int size);
void modified_huber_loss_array(float* y_hat, float* y, float threshold1, float* output, float threshold2, int size);
double random_gamma(double alpha);
void dirichlet_sample(double alpha, int n, double* sample);
float mul_array_values_by_indices(float* array, float offset, int size);

#endif
