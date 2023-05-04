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

#ifndef __IQ_H__
#define __IQ_H__

iq* init_iq(model* q_network, model** q_networks, float** states, int* actions, int* done, uint64_t size, uint64_t state_size, uint64_t batch_size, uint64_t threads,
            int feed_forward_flag, int training_mode, int adaptive_clipping_flag, int gd_flag, int lr_decay_flag, int lr_epoch_threshold,
            float momentum, float alpha1, float alpha2, float gamma, float beta1, float beta2, float beta3, float k_percentage, float adaptive_clipping_gradient_value,
            float lr, float lr_minimum, float lr_maximum, float initial_lr, float lr_decay);
void free_iqn(iq* iqn);
void train_iqn(iq* iqn, int epochs, char* directory_to_save);
#endif
