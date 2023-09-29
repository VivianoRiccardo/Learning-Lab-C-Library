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

#ifndef __ACO_PROCESS_H__
#define __ACO_PROCESS_H__

aco_struct* init_aco(int* sizes, int* widths, int* depths, int* sub_dimensions,int* activations,  int n_layers, int ants, int number_of_iterations, int time_update_best_trail, float p, float init_tau_max, float p_dec, int max_iterations_fa);
void free_aco_struct(aco_struct* s);
void update_best_trail(aco_struct* s);
void update_taus(aco_struct* s);
void set_pheromone_to_index(aco_struct* s, int index, double pheromone);
void update_path_with_pheromone(aco_struct* s, int* path, int length_path, double pheromone);
void update_pheromones(aco_struct* s);
void update_graph_pheromone_from_trail(aco_struct* s);
model* get_model_according_to_path(aco_struct* s, int index_path);
void build_path_from_ant_index(aco_struct* s, int index);
model* get_best_model(aco_struct* s);
void update_pheromones_best(aco_struct* s);
double get_stagnation(aco_struct* s);
void recompute_pheromones(aco_struct* s, double delta);
double recompute_pheromones_according_to_stagnation(aco_struct* s, double stagnation_threshold, double delta);
void calculate_pso_function_per_node(aco_struct* s);
void set_best_personal_node_fitness_pso(aco_struct* s);
void set_best_global_node_fitness_pso(aco_struct* s);
void update_pso_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations);
void update_pso_params(aco_struct* s, float current_iteration, float max_iterations);
void reset_flags_from_aco_struct(aco_struct* s);
void update_pheromones_no_min_max(aco_struct* s);
void update_graph_pheromone_from_trail_no_min_max(aco_struct* s);
void update_path_with_pheromone_no_min_max(aco_struct* s, int* path, int length_path, double pheromone);
void build_path_from_ant_index_according_to_nodes(aco_struct* s, int index);
void update_pheromones_according_to_nodes(aco_struct* s);
void update_graph_pheromone_from_trail_according_to_nodes(aco_struct* s);
void update_path_with_pheromone_according_to_nodes(aco_struct* s, int* path, int length_path, double pheromone);
void update_pheromones_best_according_to_nodes(aco_struct* s);
double recompute_pheromones_according_to_stagnation_according_to_nodes(aco_struct* s, double stagnation_threshold, double delta);
void recompute_pheromones_according_to_nodes(aco_struct* s, double delta);
double get_stagnation_according_to_nodes(aco_struct* s);
void calculate_pso_function_per_node_according_to_nodes(aco_struct* s);
void set_best_global_node_fitness_pso_according_to_nodes(aco_struct* s);
void set_pheromone_best_trail(aco_struct* s, double pheromone);
void update_pheromones_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage);
void update_graph_pheromone_from_trail_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage);
void update_path_with_pheromone_according_to_subnodes(aco_struct* s, int* path, int length_path, double pheromone, int init_node, int final_node);
void update_pheromones_best_according_to_subnodes(aco_struct* s, float init_percentage, float final_percentage);
void build_path_from_ant_index_according_to_subnodes(aco_struct* s, int index, float init_percentage, float final_percentage);
double recompute_pheromones_according_to_stagnation_according_to_subnodes(aco_struct* s, double stagnation_threshold, double delta, float init_percentage, float final_percentage);
double get_stagnation_according_to_subnodes(aco_struct* s, int init_node, int final_node);
void recompute_pheromones_according_to_subnodes(aco_struct* s, double delta, int init_node, int final_node);
model* get_model_according_to_path_debug(aco_struct* s, int index_path);
void build_best_path_from_ant_index_according_to_subnodes(aco_struct* s, int index, float init_percentage, float final_percentage);
int aco_weights_are_different(float* p1, float* p2, int size);
int get_width_according_to_node_index(aco_struct* s, int index);
float get_lambda_pso(int max_number_of_iterations, int current_iteration);
float get_lvalue_pso(int max_number_of_iterations, int current_iteration);
void update_fa_params(aco_struct* s, float current_iteration, float max_iterations);
void update_fa_nodes(aco_struct* s);
void update_gsa_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations);
void update_gsa_params(aco_struct* s, float current_iteration, float max_iterations);
void update_psogsa_params(aco_struct* s, float current_iteration, float max_iterations);
void update_psogsa_nodes(aco_struct* s, float current_iteration, float max_number_of_iterations);
void set_iteration_index_to_value(aco_struct* s, int value);
void set_x_params(aco_struct* s, float inertia_max, float inertia_min, float c1, float c2, float inertia, float v_max, float percentage_of_fireflies, float lambda_value, float beta_min,
                  float beta_zero, float softmax_temperature, float g_zero, float alpha, float omega, float rp_max, float rp_min, float alpha_velocity, float h_velocity);
void update_woa_nodes(aco_struct* s);
void update_woa_params(aco_struct* s, float current_iteration, float max_iterations);

#endif
