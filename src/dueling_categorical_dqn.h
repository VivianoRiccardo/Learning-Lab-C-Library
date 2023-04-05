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


#ifndef __DUELING_CATEGORICAL_DQN_H__
#define __DUELING_CATEGORICAL_DQN_H__
#include "llab.h"

dueling_categorical_dqn* dueling_categorical_dqn_init(int input_size, int action_size, int n_atoms, float v_min, float v_max, model* shared_hidden_layers, model* v_hidden_layers, model* a_hidden_layers, model* v_linear_last_layer, model* a_linear_last_layer);
dueling_categorical_dqn* dueling_categorical_dqn_init_without_arrays(int input_size, int action_size, int n_atoms, float v_min, float v_max, model* shared_hidden_layers, model* v_hidden_layers, model* a_hidden_layers, model* v_linear_last_layer, model* a_linear_last_layer);
void free_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void free_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn);
void free_dueling_categorical_dqn_without_arrays(dueling_categorical_dqn* dqn);
void reset_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void reset_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn);
dueling_categorical_dqn* copy_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
dueling_categorical_dqn* copy_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn);
void paste_dueling_categorical_dqn(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy);
void paste_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy);
void slow_paste_dueling_categorical_dqn(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy, float tau);
uint64_t size_of_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
uint64_t size_of_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn);
uint64_t count_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
uint64_t get_array_size_params_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
uint64_t get_array_size_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
uint64_t get_array_size_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void memcopy_vector_to_params_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void memcopy_vector_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void memcopy_params_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void memcopy_weights_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void memcopy_vector_to_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void memcopy_scores_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void set_dueling_categorical_dqn_biases_to_zero(dueling_categorical_dqn* dqn);
void set_dueling_categorical_dqn_unused_weights_to_zero(dueling_categorical_dqn* dqn);
void sum_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, dueling_categorical_dqn* input2, dueling_categorical_dqn* output);
void compare_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, dueling_categorical_dqn* input2, dueling_categorical_dqn* output);
void compare_score_dueling_categorical_dqn_with_vector(dueling_categorical_dqn* input1, float* input2, dueling_categorical_dqn* output);
void dividing_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, float value);
void reset_score_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void reinitialize_weights_according_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float percentage, float goodness);
void reinitialize_w_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
dueling_categorical_dqn* reset_edge_popup_d_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void set_low_score_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void make_the_dueling_categorical_dqn_only_for_ff(dueling_categorical_dqn* dqn);
void compute_probability_distribution(float* input , int input_size, dueling_categorical_dqn* dqn);
float* bp_dueling_categorical_network(float* input, int input_size, float* error, dueling_categorical_dqn* dqn);
float* compute_q_functions(dueling_categorical_dqn* dqn);
void compute_probability_distribution_opt(float* input , int input_size, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp);
float* bp_dueling_categorical_network_opt(float* input, int input_size, float* error, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp);
float* get_loss_for_dueling_categorical_dqn(dueling_categorical_dqn* online_net, dueling_categorical_dqn* target_net, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1);
float* get_loss_for_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1);
void save_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int n);
dueling_categorical_dqn* load_dueling_categorical_dqn(char* file);
void save_dueling_categorical_dqn_given_directory(dueling_categorical_dqn* dqn, int n, char* directory);
void set_dueling_categorical_dqn_training_edge_popup(dueling_categorical_dqn* dqn, float k_percentage);
void set_dueling_categorical_dqn_training_gd(dueling_categorical_dqn* dqn);
void set_dueling_categorical_dqn_beta(dueling_categorical_dqn* dqn, float b1, float b2);
void set_dueling_categorical_dqn_beta_adamod(dueling_categorical_dqn*  dqn, float b);
float get_beta1_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
float get_beta2_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
float get_beta3_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void set_ith_layer_training_mode_dueling_categorical_dqn_shared(dueling_categorical_dqn* dqn, int ith, int training_flag);
void set_ith_layer_training_mode_dueling_categorical_dqn_v_hid(dueling_categorical_dqn* dqn, int ith, int training_flag);
void set_ith_layer_training_mode_dueling_categorical_dqn_v_lin(dueling_categorical_dqn* dqn, int ith, int training_flag);
void set_ith_layer_training_mode_dueling_categorical_dqn_a_hid(dueling_categorical_dqn* dqn, int ith, int training_flag);
void set_ith_layer_training_mode_dueling_categorical_dqn_a_lin(dueling_categorical_dqn* dqn, int ith, int training_flag);
void set_k_percentage_of_ith_layer_dueling_categorical_dqn_shared(dueling_categorical_dqn* dqn, int ith, float k_percentage);
void set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_hid(dueling_categorical_dqn* dqn, int ith, float k_percentage);
void set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_lin(dueling_categorical_dqn* dqn, int ith, float k_percentage);
void set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_hid(dueling_categorical_dqn* dqn, int ith, float k_percentage);
void set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_lin(dueling_categorical_dqn* dqn, int ith, float k_percentage);
int get_input_layer_size_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
float* get_loss_for_dueling_categorical_dqn_opt_with_error(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1, float* new_error, float weight_error);
float compute_kl_dueling_categorical_dqn(dueling_categorical_dqn* online_net, float* state_t, float* q_functions,  float weight, float alpha, float clip);
float compute_kl_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip);
void inference_dqn(dueling_categorical_dqn* dqn);
void train_dqn(dueling_categorical_dqn* dqn);
void dueling_dqn_eliminate_noisy_layers(dueling_categorical_dqn* dqn);
void assign_noise_arrays_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float** noise_biases1, float** noise1,float** noise_biases2, float** noise2,float** noise_biases3, float** noise3,float** noise_biases4, float** noise4,float** noise_biases5, float** noise5);
void reinitialize_weights_according_to_scores_and_inner_info_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void memcopy_vector_to_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int* vector);
void memcopy_indices_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int* vector);
void free_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void free_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void assign_vector_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector);
void set_null_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
void set_null_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn);
float compute_l1_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip);
float compute_l1_dueling_categorical_dqn(dueling_categorical_dqn* online_net, float* state_t, float* q_functions,  float weight, float alpha, float clip);


#endif
