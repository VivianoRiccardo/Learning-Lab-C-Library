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
void paste_model(model* m, model* copy);
void paste_w_model(model* m, model* copy);
void slow_paste_model(model* m, model* copy, float tau);
model* reset_model(model* m);
model* reset_model_except_partial_derivatives(model* m);
model* reset_model_without_dwdb(model* m);
model* reset_model_for_edge_popup(model* m);
uint64_t size_of_model(model* m);
void save_model(model* m, int n);
void save_model_given_directory(model* m, int n, char* directory);
model* load_model(char* file);
model* load_model_with_file_already_opened(FILE* fr);
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
uint64_t count_weights(model* m);
uint64_t get_array_size_params_model(model* f);
uint64_t get_array_size_weights_model(model* f);
uint64_t get_array_size_scores_model(model* f);
void memcopy_vector_to_params_model(model* f, float* vector);
void memcopy_vector_to_weights_model(model* f, float* vector);
void memcopy_vector_to_scores_model(model* f, float* vector);
void memcopy_params_to_vector_model(model* f, float* vector);
void memcopy_weights_to_vector_model(model* f, float* vector);
void memcopy_scores_to_vector_model(model* f, float* vector);
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
void set_model_biases_to_zero(model* m);
void set_model_unused_weights_to_zero(model* m);
void set_model_training_edge_popup(model* m, float k_percentage);
void set_model_training_gd(model* m);
void sum_score_model(model* input1, model* input2, model* output);
void compare_score_model(model* input1, model* input2, model* output);
void compare_score_model_with_vector(model* input1, float* input2, model* output);
void dividing_score_model(model* m, float value);
void avaraging_score_model(model* avarage, model** m, int n_model);
void reset_score_model(model* f);
void reinitialize_weights_according_to_scores_model(model* m, float percentage, float goodness);
void reinitialize_w_model(model* m);
model* reset_edge_popup_d_model(model* m);
int check_model_last_layer(model* m);
void set_low_score_model(model* f);
void free_model_without_learning_parameters(model* m);
model* copy_model_without_learning_parameters(model* m);
void paste_model_without_learning_parameters(model* m, model* copy);
model* reset_model_without_learning_parameters(model* m);
model* reset_model_without_dwdb_without_learning_parameters(model* m);
uint64_t size_of_model_without_learning_parameters(model* m);
void ff_fcl_fcl_without_learning_parameters(fcl* f1, fcl* f2, fcl* f3);
void ff_fcl_cl_without_learning_parameters(fcl* f1, cl* f2, cl* f3);
void ff_cl_fcl_without_learning_parameters(cl* f1, fcl* f2, fcl* f3);
void ff_cl_cl_without_learning_parameters(cl* f1, cl* f2, cl* f3);
float* bp_fcl_fcl_without_learning_parameters(fcl* f1, fcl* f2, fcl* f3, float* error);
float* bp_fcl_cl_without_learning_parameters(fcl* f1, cl* f2,cl* f3, float* error);
float* bp_cl_cl_without_learning_parameters(cl* f1, cl* f2,cl* f3, float* error);
float* bp_cl_fcl_without_learning_parameters(cl* f1, fcl* f2,fcl* f3, float* error);
void model_tensor_input_ff_without_learning_parameters(model* m, model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input);
float* model_tensor_input_bp_without_learning_parameters(model* m, model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension);
float* ff_error_bp_model_once_opt(model* m,model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input, float* output);
void free_model_without_arrays(model* m);
void make_the_model_only_for_ff(model* m);
float* get_output_layer_from_model(model* m);
int get_output_dimension_from_model(model* m);
void set_model_beta(model* m, float beta1, float beta2);
float get_beta1_from_model(model* m);
float get_beta2_from_model(model* m);
void set_ith_layer_training_mode_model(model* m, int ith, int training_flag);
void set_k_percentage_of_ith_layer_model(model* m, int ith, float k_percentage);
void set_model_beta_adamod(model* m, float beta);
float get_beta3_from_model(model* m);
int get_input_layer_size(model* m);
int ff_fcl_fcl_without_arrays(fcl* f1, fcl* f2);
int ff_fcl_cl_without_arrays(fcl* f1, cl* f2);
int ff_cl_fcl_without_arrays(cl* f1, fcl* f2);
int ff_cl_cl_without_arrays(cl* f1, cl* f2);
int model_tensor_input_ff_without_arrays(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input);
void inference_model(model* m);
void train_model(model* m);
void model_eliminate_noisy_layers(model* m);
void assign_noise_arrays_model(model* m, float** noise_biases, float** noise);
void reinitialize_weights_according_to_scores_and_inner_info_model(model* m);
void memcopy_vector_to_indices_model(model* f, int* vector);
void memcopy_indices_to_vector_model(model* f, int* vector);
void free_scores_model(model* m);
void free_indices_model(model* m);
void assign_vector_to_scores_model(model* f, float* vector);
void set_null_scores_model(model* m);
void set_null_indices_model(model* m);
void reinitialize_weights_according_to_scores_model_only_percentage(model* m, float percentage);
void memcopy_vector_to_indices_model2(model* f, int* vector);
void inverse_q_model_error(model* m, float* output);
void policy_gradient_model_error(model* m, float* output);
model* reset_model_only_for_ff(model* m);
void mse_model_error_only_for_ff(model* m, float* output);
void cross_entropy_model_error_only_for_ff(model* m, float* output);
void focal_model_error_only_for_ff(model* m, float* output);
void huber_one_model_error_only_for_ff(model* m, float* output);
void huber_two_model_error_only_for_ff(model* m, float* output);
void kl_model_error_only_for_ff(model* m, float* output);
void entropy_model_error_only_for_ff(model* m, float* output);
void compute_model_error_only_for_ff(model* m, float* output);

#endif
