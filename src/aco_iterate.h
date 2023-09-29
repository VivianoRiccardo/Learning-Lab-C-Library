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

#ifndef __ACO_ITERATE_H__
#define __ACO_ITERATE_H__

void aco_tracker_execute_operation(aco_tracker* t);
void aco_tracker_build_model(aco_tracker* t);
int aco_tracker_next(aco_tracker* t);
int aco_tracker_next_weithout_setting_flags(aco_tracker* t);
int aco_tracker_next_by_index(aco_tracker* t, int index);
int update_aco_tracker_next_according_to_path(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min);
int aco_tracker_best_next(aco_tracker* t);
model* build_model_from_tracker(aco_tracker* t);
void reset_flags(aco_node* root);
void get_all_nodes_and_edges(aco_node* root, aco_node*** nodes, aco_edge*** edges, int* n_nodes, int* n_edges);
void compute_all_nodes_edges(aco_node* root, aco_node*** nodes, aco_edge*** edges, int* n_nodes, int* n_edges);
void aco_tracker_build_model_complete(aco_tracker* t);
void aco_tracker_build_model_complete2(aco_tracker* t);
int aco_tracker_best_next_set_flag(aco_tracker* t);
void sign_best(aco_struct* s);
int update_aco_tracker_next_according_to_path_best(aco_tracker* t, int* path, int path_index, int path_size);
int update_aco_tracker_next_according_to_path_no_min_max(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min);
int aco_tracker_next_weithout_setting_flags_according_to_nodes(aco_tracker* t);
int update_aco_tracker_next_according_to_path_according_to_nodes(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min);
int update_aco_tracker_next_according_to_path_according_to_subnodes(aco_tracker* t,double pheromone, int* path, int path_index, int path_size, float p, double t_max, double t_min, int init_node, int final_node);
int aco_tracker_next_weithout_setting_flags_according_to_best(aco_tracker* t);

#endif
