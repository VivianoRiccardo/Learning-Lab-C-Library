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

#ifndef __EFFICIENTZEROMODEL_H__
#define __EFFICIENTZEROMODEL_H__


efficientzeromodel* init_efficientzero_model(model* rapresentation_h, model* dynamics_g, model* prediction_f, model* prediction_f_policy,
                                             model* prediction_f_value, model* reward_prediction_model, rmodel* reward_prediction_rmodel,
                                             model* reward_prediction_temporal_model, model* p1, model* p2, int threads, int lstm_window);
void make_efficientzeromodel_only_for_ff(efficientzeromodel* m);
void free_efficientzero_model(efficientzeromodel* m);
void efficientzero_ff_p1(efficientzeromodel* m, float* input);
float* efficientzero_bp_p1(efficientzeromodel* m, float* input, float* error);
void efficientzero_reset_p1(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_p1(efficientzeromodel* m);
void efficientzero_ff_p2(efficientzeromodel* m, float* input);
float* efficientzero_bp_p2(efficientzeromodel* m, float* input, float* error);
void efficientzero_reset_p2(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_p2(efficientzeromodel* m);
void efficientzero_ff_rapresentation_h(efficientzeromodel* m, float* input);
float* efficientzero_bp_rapresentation_h(efficientzeromodel* m, float* input, float* error);
void efficientzero_reset_rapresentation_h(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_rapresentation_h(efficientzeromodel* m);
void efficientzero_ff_dynamics_g(efficientzeromodel* m, float* input);
float* efficientzero_bp_dynamics_g(efficientzeromodel* m, float* input, float* error);
void efficientzero_reset_dynamics_g(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_dynamics_g(efficientzeromodel* m);
void efficientzero_ff_prediction_f(efficientzeromodel* m, float* input);
float* efficientzero_bp_prediction_f(efficientzeromodel* m, float* input, float* error);
void efficientzero_reset_prediction_f(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_prediction_f(efficientzeromodel* m);
void efficientzero_ff_reward_prediction(efficientzeromodel* m, float** inputs);
float** efficientzero_bp_reward_prediction(efficientzeromodel* m, float** inputs, float** errors);
void efficientzero_reset_reward_prediction(efficientzeromodel* m);
void efficientzero_reset_only_for_ff_reward_prediction(efficientzeromodel* m);
void efficientzero_ff_p1_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_p1_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_p1_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_p1_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_ff_p2_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_p2_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_p2_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_p2_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_ff_rapresentation_h_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_rapresentation_h_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_rapresentation_h_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_rapresentation_h_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_ff_dynamics_g_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_dynamics_g_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_dynamics_g_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_dynamics_g_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_ff_prediction_f_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_prediction_f_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_prediction_f_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_prediction_f_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_ff_reward_prediction_opt(efficientzeromodel* m, float** inputs, int threads);
float** efficientzero_bp_reward_prediction_opt(efficientzeromodel* m, float** inputs, float** errors, int threads);
void efficientzero_reset_reward_prediction_without_learning_parameters(efficientzeromodel* m, int threads);
void efficientzero_reset_only_for_ff_reward_prediction_without_learning_parameters(efficientzeromodel* m, int threads);
void save_efficientzero_model(efficientzeromodel* m, int n);
efficientzeromodel* load_efficientzeromodel(char* file, int batch_size);
void save_efficientzero_model_given_directory(efficientzeromodel* m, int n, char* directory);
void reset_efficientzeromodel(efficientzeromodel* m);
void reset_efficientzeromodel_only_for_ff(efficientzeromodel* m);
void reset_efficientzeromodel_without_learning_parameters(efficientzeromodel* m);
void reset_efficientzeromodel_only_for_ff_without_learning_parameters(efficientzeromodel* m);
efficientzeromodel* copy_efficientzero_model(efficientzeromodel* m);
void paste_efficientzero_model(efficientzeromodel* m, efficientzeromodel* copy);
void slow_paste_efficientzero_model(efficientzeromodel* m, efficientzeromodel* copy, float tau);
uint64_t size_of_efficientzero_model(efficientzeromodel* m);
uint64_t size_of_efficientzero_model_without_learning_parameters(efficientzeromodel* m);
uint64_t count_weights_efficientzero_model(efficientzeromodel* m);
uint64_t get_array_size_params_efficientzero_model(efficientzeromodel* m);
uint64_t get_array_size_scores_efficientzero_model(efficientzeromodel* m);
uint64_t get_array_size_weights_efficientzero_model(efficientzeromodel* m);
void memcopy_vector_to_params_efficientzero_model(efficientzeromodel* m, float* vector);
void memcopy_vector_to_scores_efficientzero_model(efficientzeromodel* m, float* vector);
void memcopy_params_to_vector_efficientzero_model(efficientzeromodel* m, float* vector);
void memcopy_weights_to_vector_efficientzero_model(efficientzeromodel* m, float* vector);
void memcopy_vector_to_weights_efficientzero_model(efficientzeromodel* m, float* vector);
void memcopy_scores_to_vector_efficientzero_model(efficientzeromodel* m, float* vector);
void set_efficientzero_model_beta(efficientzeromodel* m, float b1, float b2);
void set_efficientzero_model_beta_adamond(efficientzeromodel* m, float b1);
void efficientzero_ff_reward_prediction_single_cell(efficientzeromodel* m, float* inputs, float** hidden_states, float** cell_states, float** new_hidden_states, float** new_cell_states, int fullfill_flag);
void efficientzero_reset_only_for_ff_reward_prediction_single_cell(efficientzeromodel* m);
void set_efficientzero_model_training_edge_popup(efficientzeromodel* m, float k_percentage);

#endif
