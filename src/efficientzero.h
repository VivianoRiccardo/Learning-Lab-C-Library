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

#ifndef __EFFICIENTZERO_H__
#define __EFFICIENTZERO_H__

efficientzero* init_efficientzero(int sampling_flag, int gd_flag, int lr_decay_flag, int feed_forward_flag, int training_mode, int clipping_flag, int adaptive_clipping_flag, int batch_size,int threads, 
                      uint64_t epochs_to_copy_target, uint64_t max_buffer_size, uint64_t n_step_rewards, uint64_t lr_epoch_threshold,
                      float alpha_priorization, float beta_priorization, float gamma, float lambda_value, float tau_copying, float beta1, float beta2,
                      float beta3, float k_percentage, float clipping_gradient_value, float adaptive_clipping_gradient_value, float lr, float lr_minimum, float lr_maximum, float lr_decay, float momentum,
                      float beta_priorization_increase, efficientzeromodel* m);
void free_efficientzero(efficientzero* r);
void add_buffer_state_efficientzero(efficientzero* r, uint index);
void add_buffer_state_reward_sampling_efficientzero(efficientzero* r, uint index);
void update_buffer_state_efficientzero(efficientzero* r, uint index, float error);
void update_buffer_state_reward_sampling_efficientzero(efficientzero* r, uint index, float previous_reward);
void add_experience_efficientzero(efficientzero* r, float* state_t, float* state_t_1, float value, int action, float reward, int nonterminal_s_t_1);

#endif
