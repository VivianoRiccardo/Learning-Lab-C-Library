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

#ifndef __MULTI_CORE_DUELING_CATEGORICAL_DQN_H__
#define __MULTI_CORE_DUELING_CATEGORICAL_DQN_H__

void* dueling_categorical_dqn_train_thread(void* _args);
void dueling_categorical_dqn_train(int threads, dueling_categorical_dqn* online_net,dueling_categorical_dqn* target_net, dueling_categorical_dqn** online_net_wlp, dueling_categorical_dqn** target_net_wlp, float** states_t, float* rewards_t, int* actions_t, float** states_t_1, int* nonterminals_t_1, float lambda_value, int state_sizes);
void* dueling_categorical_dqn_thread_sum(void* _args);
dueling_categorical_dqn* sum_dueling_categorical_dqn_partial_derivatives_multithread(dueling_categorical_dqn** batch_m, dueling_categorical_dqn* m, int n, int depth);
void dueling_categorical_reset_without_learning_parameters_reset(dueling_categorical_dqn** dqn, int threads);
void* dueling_categorical_dqn_reset_thread(void* _args);

#endif
