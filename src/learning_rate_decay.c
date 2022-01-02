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

#include "llab.h"

// whenever you want
void constant_decay(float* lr, float decay, float minimum){
    if((*lr <= minimum))
        return;
    lr[0]-=decay;
}

// after each epoch usually
void time_based_decay(float* lr, float decay, float minimum, int iterations){
    if((*lr <= minimum))
        return;
    lr[0]*=(1.0 / (1.0 + decay * (float)(iterations)));
}

// usually done after epochs_drop times
void step_decay(float* lr, float initial_lr, float drop, float minimum, int epoch, int epochs_drop){
    lr[0] = initial_lr * pow((double)drop,floor((double)((1+epoch)/epochs_drop)));
}

void cosine_annealing(float* lr, float lr_minimum, float lr_maximum, int epoch, int epoch_threshold){
    lr[0] = lr_minimum + (float)(lr_maximum-lr_minimum)*(1.0+cos((double)(1+PI*(float)(epoch/epoch_threshold))));
}


void update_lr(float* lr, float lr_minimum, float lr_maximum,float initial_lr, float decay, int epoch, int epoch_threshold, int lr_decay_flag){
    if(lr_decay_flag == LR_NO_DECAY)
        return;
    if(epoch_threshold >= epoch)
        if (epoch_threshold%epoch)
            return;
    else
        if(epoch%epoch_threshold)
            return;
            
    if (lr_decay_flag == LR_CONSTANT_DECAY)
        constant_decay(lr,decay,lr_minimum);
    else if(lr_decay_flag == LR_TIME_BASED_DECAY)
        time_based_decay(lr,decay,lr_minimum,epoch);
    else if(lr_decay_flag == LR_STEP_DECAY)
        step_decay(lr,initial_lr,decay,lr_minimum,epoch,epoch_threshold);
    else if(lr_decay_flag == LR_CONSTANT_DECAY)
        cosine_annealing(lr,lr_minimum,lr_maximum,epoch,epoch_threshold);
    
}
