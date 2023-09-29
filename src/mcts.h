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

#ifndef __MCTS_H__
#define __MCTS_H__

void free_mcts(mcts* t);
void free_mcts_node(mcts_node* n);
float* get_mcts_probability(mcts* t, float temperature);
double get_mcts_v(mcts* t, float* p);
double get_sum_visited_children_q_normalized_mcts_node(mcts* t, mcts_node* n);
double get_visited_children_q_mcts_node(mcts_node* n);
mcts* init_mcts(efficientzeromodel* m, float* init_state, double value_offset, double reward_offset, double gamma_reward, double dirichlet_alpha, double c_init, double c_base, double noise_epsilon, uint64_t init_state_size, uint64_t maximum_depth);
mcts_edge* init_mcts_edge(mcts_node* input_node, mcts_node* output_node, float prior_probability);
mcts_node* init_mcts_node(float* state, float q_value, float reward, float v, uint64_t state_size, uint64_t depth, uint64_t visit_count, uint64_t n_edges, uint64_t lstm_layers, uint64_t* h_states_size);
void mcts_node_add_edge(mcts_node* n, mcts_edge* e);
void mcts_node_set_state(mcts_node* n, float* state, uint64_t state_size);
int node_is_full_of_edges(mcts_node* n);
double visit_node(mcts* t, mcts_node* n, double q_hat_parent, uint64_t depth);

#endif
