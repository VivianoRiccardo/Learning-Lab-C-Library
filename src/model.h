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

#ifndef __MODEL_H__
#define __MODEL_H__

model* network(int layers, int n_rl, int n_cl, int n_fcl, rl** rls, cl** cls, fcl** fcls);
void free_model(model* m);
model* copy_model(model* m);
void save_model(model* m, int n);
void heavy_save_model(model* m, int n);
model* load_model(char* file);
model* heavy_load_model(char* file);
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
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void sum_model_partial_derivatives(model* m, model* m2, model* m3);
unsigned long long int size_of_model(model* m);
void paste_model(model* m, model* copy);
int count_weights(model* m);
void slow_paste_model(model* m, model* copy, float tau);
int get_array_size_params_model(model* f);
void memcopy_vector_to_params_model(model* f, float* vector);
void memcopy_params_to_vector_model(model* f, float* vector);
void memcopy_vector_to_derivative_params_model(model* f, float* vector);
void memcopy_derivative_params_to_vector_model(model* f, float* vector);
void set_model_error(model* m, int error_flag, float threshold1, float threshold2, float gamma, float* alpha, int output_dimension);
void mse_model_error(model* m, float* output);
void cross_entropy_model_error(model* m, float* output);
void focal_model_error(model* m, float* output);
void huber_one_model_error(model* m, float* output);
void huber_two_model_error(model* m, float* output);
void kl_model_error(model* m, float* output);
void entropy_model_error(model* m, float* output);
void compute_model_error(model* m, float* output);
float* ff_error_bp_model_once(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* output);
void sum_models_partial_derivatives(model* sum_m, model** models, int n_models);
void set_model_biases_to_zero(model* m);
void set_model_training_edge_popup(model* m, float k_percentage);
void set_model_training_gd(model* m);
void set_model_unused_weights_to_zero(model* m);
int adjust_model_weights_after_edge_popup(model* m);
void get_subnetwork_from_edge_popup(model* m);
void paste_w_model(model* m, model* copy);
void sum_score_model(model* input1, model* input2, model* output);
void dividing_score_model(model* m, float value);
void avaraging_score_model(model* avarage, model** m, int n_model);

#endif
