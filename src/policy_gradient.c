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



policy_gradient* init_policy_gradient(model* m, model** ms, uint64_t batch_size, uint64_t threads,
            int feed_forward_flag, int training_mode, int adaptive_clipping_flag, int gd_flag, int lr_decay_flag, int lr_epoch_threshold,
            float momentum, float beta1, float beta2, float beta3, float k_percentage, float adaptive_clipping_gradient_value,
            float lr, float lr_minimum, float lr_maximum, float lr_decay, float softmax_temperature, int entropy_flag, float entropy_alpha, int dde_flag, float dde_alpha){
                
    if(m == NULL){
        fprintf(stderr,"Error: you passed a model as null!\n");
        exit(1);
    }
    
    if(ms == NULL && threads > 1){
        fprintf(stderr,"Error: your threads are higher than 1 but u passed null as threaded models!\n");
        exit(1);
    }
    
    if (threads > batch_size){
        threads = batch_size;
    }
    
    if(feed_forward_flag != EDGE_POPUP || feed_forward_flag != FULLY_FEED_FORWARD){
        fprintf(stderr,"Error: feed_forward_flag not recognized!\n");
        exit(1);
    }
    
    if(training_mode != FREEZE_BIASES || training_mode != EDGE_POPUP || training_mode != GRADIENT_DESCENT){
        fprintf(stderr,"Error: training_mode not recognized!\n");
        exit(1);
    }
    
    if(gd_flag != NESTEROV || gd_flag != ADAM || gd_flag != ADAMOD || gd_flag != RADAM || gd_flag != DIFF_GRAD){
        fprintf(stderr,"Error: gradient_Descent_flag not recognized!\n");
        exit(1);
    }
    
    if((training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP ) && (k_percentage <= 0 || k_percentage >= 1)){
        fprintf(stderr,"Error: somenthing is wrong with your k_percentage setup!\n");
        exit(1);
    }
    
    if(lr_minimum > lr_maximum){
        fprintf(stderr,"Error: your lr_minimum can't be > lr_maximum!\n");
        exit(1);
    }
    
    policy_gradient* pg = (policy_gradient*)malloc(sizeof(policy_gradient));
    pg->m = m;
    pg->ms = ms; 
    pg->batch_size = batch_size;
    pg->threads = threads;
    pg->feed_forward_flag = feed_forward_flag;
    pg->training_mode = training_mode;
    pg->adaptive_clipping_flag = adaptive_clipping_flag;
    pg->gd_flag = gd_flag;
    pg->lr_decay_flag = lr_decay_flag;
    pg->lr_epoch_threshold = lr_epoch_threshold;
    pg->momentum = momentum;
    pg->beta1 = beta1;
    pg->beta2 = beta2;
    pg->beta3 = beta3;
    pg->k_percentage = k_percentage;
    pg->adaptive_clipping_flag = adaptive_clipping_flag;
    pg->lr = lr;
    pg->lr_minimum = lr_minimum;
    pg->lr_maximum = lr_maximum;
    pg->initial_lr = lr;
    pg->lr_decay = lr_decay;
    pg->softmax_temperature = softmax_temperature;
    pg->entropy_flag = entropy_flag;
    pg->dde_flag = dde_flag;
    pg->dde_alpha = dde_alpha;
    float* rewards = (float*)calloc(pg->m->output_dimension, sizeof(float));
    set_model_error(pg->m,POLICY_GRADIENT,((float)pg->entropy_flag)*pg->entropy_alpha, ((float)pg->dde_flag)*pg->dde_alpha,pg->softmax_temperature,rewards, pg->m->output_dimension);
    set_model_beta(pg->m,beta1,beta2);
    set_model_beta_adamod(pg->m,beta3);
    if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
        set_model_training_edge_popup(pg->m,k_percentage);
    }
    else{
        set_model_training_gd(pg->m);
    }
    int i;
    if(threads > 1){
        for(i = 0; i < threads; i++){
            if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
                set_model_training_edge_popup(pg->ms[i],k_percentage);
            }
            else{
                set_model_training_gd(pg->ms[i]);
            }
            set_model_error(pg->ms[i],POLICY_GRADIENT,((float)pg->entropy_flag)*pg->entropy_alpha, ((float)pg->dde_flag)*pg->dde_alpha,pg->softmax_temperature,rewards, pg->m->output_dimension);
        }
    }
    free(rewards);
}

void policy_gradient_training_passage(policy_gradient* pg, float** states, float* rewards, float** states_for_dde){
    int i,j, k, min = pg->threads, input_size = get_input_layer_size(pg->m);
    for(i = 0; i < pg->batch_size; i+=pg->threads){
		min = pg->threads;
		if(pg->batch_size - i < pg->threads)
			min = pg->batch_size - i;
		if(pg->threads == 1){
			for(j = 0; j < pg->m->output_dimension; j++){
				pg->m->error_alpha[j] = rewards[i];
			}
			ff_error_bp_model_once(pg->m,1,1,input_size,states[i],states_for_dde[i]);
			reset_model_except_partial_derivatives(pg->m);
		}
		else{
			for(k = 0; k < min; k++){
				for(j = 0; j < pg->m->output_dimension; j++){
					pg->ms[k]->error_alpha[j] = rewards[i];
				}
			}
			ff_error_bp_model_multicore_opt(pg->ms,pg->m,1,1,input_size,states+i,min, min, states_for_dde+i,NULL);
			sum_models_partial_derivatives_multithread(pg->ms,pg->m,min,0);
			for(k = 0; k < min; k++){
				reset_model_without_learning_parameters(pg->ms[k]);
			}
		}
	}
} 
            
