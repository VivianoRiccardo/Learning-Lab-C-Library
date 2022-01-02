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

#ifndef __LEARNING_RATE_DECAY_H__
#define __LEARNING_RATE_DECAY_H__

void constant_decay(float* lr, float decay, float minimum);
void time_based_decay(float* lr, float decay, float minimum, int iterations);
void step_decay(float* lr, float initial_lr, float drop, float minimum, int epoch, int epochs_drop);
void cosine_annealing(float* lr, float lr_minimum, float lr_maximum, int epoch, int epoch_threshold);
void update_lr(float* lr, float lr_minimum, float lr_maximum,float initial_lr, float decay, int epoch, int epoch_threshold, int lr_decay_flag);

#endif
