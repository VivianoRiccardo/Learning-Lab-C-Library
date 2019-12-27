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

/* This function initializes a oustrategy structure*/
oustrategy* init_oustrategy(int action_dim, float* act_max, float* act_min){
    oustrategy* ou = (oustrategy*)malloc(sizeof(oustrategy));
    ou->mu = 0;
    ou->theta = 0.15;
    ou->max_sigma = 0.3;
    ou->min_sigma = 0.3;
    ou->sigma = 0.3;
    ou->decay_period = 100000;
    ou->action_dim = action_dim;
    ou->action_space = (float*)calloc(action_dim,sizeof(float));
    ou->state = (float*)calloc(action_dim,sizeof(float));
    ou->action_max = act_max;
    ou->action_min = act_min;
    return ou;
}


/* This function frees a space allocated by a oustrategy structure*/
void free_oustrategy(oustrategy* ou){
    free(ou->action_space);
    free(ou->state);
    free(ou);
}

/* takes the action space and sets it to mu*/
void reset_oustrategy(oustrategy* ou){
    int i;
    for(i = 0; i < ou->action_dim; i++){
        ou->action_space[i] = ou->mu;
    }
}

/* this function evolve a oustrategy struct state*/
void evolve_state(oustrategy* ou){
    float* dx = (float*)malloc(sizeof(float)*ou->action_dim);
    int i;
    for(i = 0; i < ou->action_dim; i++){
        dx[i] = ou->theta*(ou->mu-ou->state[i])+ou->sigma*random_normal();
        ou->state[i]+=dx[i];
    }
    
    free(dx);
}

// if you don't know t, set to 0
void get_action(oustrategy* ou, long long unsigned int t, float* actions){
    int i;
    evolve_state(ou);
    float min;
    if(1 < (double)(t/ou->decay_period))
        min = 1;
    else
        min = (float)((t/ou->decay_period));
    ou->sigma = ou->max_sigma - (ou->max_sigma - ou->min_sigma) * min;
    for(i = 0; i < ou->action_dim; i++){
        actions[i]+=ou->state[i];
        if(actions[i] > ou->action_max[i])
            actions[i] = ou->action_max[i];
        if(actions[i] < ou->action_min[i])
            actions[i] = ou->action_min[i];
    }
    
}
