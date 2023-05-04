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



iq* init_iq(model* q_network, model** q_networks, float** states, int* actions, int* done, uint64_t size, uint64_t state_size, uint64_t batch_size, uint64_t threads,
            int feed_forward_flag, int training_mode, int adaptive_clipping_flag, int gd_flag, int lr_decay_flag, int lr_epoch_threshold,
            float momentum, float alpha1, float alpha2, float gamma, float beta1, float beta2, float beta3, float k_percentage, float adaptive_clipping_gradient_value,
            float lr, float lr_minimum, float lr_maximum, float initial_lr, float lr_decay){
            
            if(q_network == NULL){
                fprintf(stderr,"Error you passed a q_network that is NULL!\n");
                exit(1);
            }
            
            if(q_networks == NULL && threads > 1){
                fprintf(stderr,"Error: you passed q_networks = NULL and set threads > 1!\n");
                exit(1);
            }
            
            if(states == NULL){
                fprintf(stderr,"Error: you passed states as null!\n");
                exit(1);
            }    
            
            if(actions == NULL){
                fprintf(stderr,"Error: you passed actions as null!\n");
                exit(1);
            }
            
            if(done == NULL){
                fprintf(stderr,"Error: you passed done as NULL!\n");
                exit(1);
            }
            
            if(!size || !state_size || !batch_size || !threads){
                fprintf(stderr,"Error: one of the following elements is set to 0: uint size, uint state_size, uint batch_size, uint threads\n");
                exit(1);
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
            
            if(get_input_layer_size(q_network) != state_size){
                fprintf(stderr,"Error: your state size does not match the input layer of your model!\n");
                exit(1);
            }
            
            if(threads > batch_size)
                threads = batch_size;
            
            if(size%batch_size){
                fprintf(stderr,"Error: you should have batch size perfectly dividing the size!\n");
                exit(1);
            }
            
            iq* iqn = (iq*)malloc(sizeof(iq));
            iqn->states = states;
            iqn->action_t = actions;
            iqn->done = done;
            iqn->size = size;
            iqn->state_size = state_size;
            iqn->batch_size = batch_size;
            iqn->threads = threads;
            iqn->feed_forward_flag = feed_forward_flag;
            iqn->training_mode = training_mode;
            iqn->adaptive_clipping_flag = adaptive_clipping_flag;
            iqn->gd_flag = gd_flag;
            iqn->lr_decay_flag = lr_decay_flag;
            iqn->momentum = momentum;
            iqn->alpha1 = alpha1;
            iqn->alpha2 = alpha2;
            iqn->gamma = gamma;
            iqn->beta1 = beta1;
            iqn->beta2 = beta2;
            iqn->beta3 = beta3;
            iqn->k_percentage = k_percentage;
            iqn->adaptive_clipping_gradient_value = adaptive_clipping_gradient_value;
            iqn->lr = lr;
            iqn->lr_minimum = lr_minimum;
            iqn->lr_maximum = lr_maximum;
            iqn->initial_lr = initial_lr;
            iqn->lr_epoch_threshold = lr_epoch_threshold;
            iqn->lr_decay = lr_decay;
            iqn->q_network = q_network;
            iqn->q_networks = q_networks;
            iqn->output_actions = (float*)calloc(get_output_dimension_from_model(q_network), sizeof(float));
            
            int i;
            iqn->index = (int*)calloc(size,sizeof(int));
            for(i = 0; i < size; i++){
                iqn->index[i] = i;
            }
            if(threads == 1)
                set_model_error(iqn->q_network,INVERSE_Q_LEARNING,alpha1,alpha2,gamma,iqn->output_actions,get_output_dimension_from_model(q_network));
            else{
                for(i = 0; i < threads; i++){
                    set_model_error(iqn->q_networks[i],INVERSE_Q_LEARNING,alpha1,alpha2,gamma,iqn->output_actions,get_output_dimension_from_model(q_network));
                } 
            }
            
            set_model_beta(iqn->q_network,beta1,beta2);
            set_model_beta_adamod(iqn->q_network,beta3);
            if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
                set_model_training_edge_popup(iqn->q_network,k_percentage);
            }
            else{
                set_model_training_gd(iqn->q_network);
            }
            
            return iqn;
}


void free_iqn(iq* iqn){
    free(iqn->index);
    free(iqn->output_actions);
    free(iqn);
}

void train_iqn(iq* iqn, int epochs, char* directory_to_save){
    int i,j,k,z,w, min = 0, count, iter = 1;
    model** models_to_handle = NULL;
    float** next_q = (float**)malloc(sizeof(float*)*iqn->batch_size);
    float** inputs = (float**)malloc(sizeof(float*)*iqn->batch_size);
    for(i = 0; i < iqn->batch_size; i++){
        next_q[i] = (float*)calloc(get_output_dimension_from_model(iqn->q_network), sizeof(float));
    }
    
    if(iqn->threads > 1)
         models_to_handle = (model**)malloc(sizeof(model*)*iqn->threads);
         
    for(i = 0; i < epochs; i++){
        shuffle_int_array(iqn->index,iqn->size);
        for(j = 0; j < iqn->size; j+=iqn->batch_size){
            
            // computing next_q
            for(k = j; k < iqn->size && k < j+iqn->batch_size; k += min){
                min = iqn->threads;
                if(iqn->size - 1 - k < min)
                    min = iqn->size - 1 - k;
                if(j+iqn->batch_size - 1 - k < min)
                    min = j+iqn->batch_size - 1 - k;
                    
                if(iqn->threads == 1){
                    if(iqn->index[k] != iqn->size-1 && !iqn->done[iqn->index[k]]){
                        model_tensor_input_ff(iqn->q_network,1,1,iqn->state_size,iqn->states[iqn->index[k]+1]);
                        copy_array(iqn->q_network->output_layer,next_q[k-j], get_output_dimension_from_model(iqn->q_network));
                        reset_model(iqn->q_network);// create reset only for feedforward (faster)
                    }
                }
                
                else{
                    for(count = 0, z = k; z < k + min; z++){
                        if(!iqn->done[iqn->index[z]] && iqn->index[z] != iqn->size-1){
                            models_to_handle[count] = iqn->q_networks[z-k];// models_to_handle could be removed
                            inputs[count] = iqn->states[iqn->index[z]+1];
                            count++;
                        }
                    }
                    model_tensor_input_ff_multicore_opt(models_to_handle, iqn->q_network, 1, 1, iqn->state_size, inputs, count, count);
                    for(count = 0, z = k; z < k + min; z++){
                        if(!iqn->done[iqn->index[z]] && iqn->index[z] != iqn->size-1){
                            copy_array(models_to_handle[count]->output_layer, next_q[z-j], get_output_dimension_from_model(iqn->q_network));
                            reset_model_without_learning_parameters(models_to_handle[count]);// create reset only for feedforward (faster)
                            count++;
                        }
                    }
                }
            }
            
            // training
            for(k = j; k < iqn->size && k < j+iqn->batch_size; k += min){
                min = iqn->threads;
                if(iqn->size - 1 - k < min)
                    min = iqn->size-1-k;
                if(j+iqn->batch_size - 1 - k < min)
                    min = j+iqn->batch_size - 1 - k;
            
                if(iqn->threads == 1){
                    // giving the output q function according to the action taken
                    for(z = 0; z < get_output_dimension_from_model(iqn->q_network); z++){
                        iqn->q_network->error_alpha[z] = 0;
                        if(z == (int)iqn->action_t[iqn->index[k]])
                            iqn->q_network->error_alpha[z] = 1;
                    }
                    // setting gamma to 0 if there is no next state for the current state we r looking at
                    if(iqn->index[k] == iqn->size-1 || iqn->done[iqn->index[k]])
                        iqn->q_network->error_gamma = 0;
                    else
                        iqn->q_network->error_gamma = iqn->gamma;
                    ff_error_bp_model_once(iqn->q_network,1,1,iqn->state_size,iqn->states[iqn->index[k]], next_q[k-j]);
                    reset_model_except_partial_derivatives(iqn->q_network);
                }
                else{
                    for(z = k; z < k + min; z++){
                        for(w = 0; w < get_output_dimension_from_model(iqn->q_network); w++){
                            iqn->q_networks[z-k]->error_alpha[w] = 0;
                            if(w == (int)iqn->action_t[iqn->index[z]])
                                iqn->q_networks[z-k]->error_alpha[w] = 1;
                            
                        }
                        if(iqn->done[iqn->index[z]] || iqn->index[z] == iqn->size-1)
                            iqn->q_networks[z-k]->error_gamma = 0;
                        else
                            iqn->q_networks[z-k]->error_gamma = iqn->gamma;
                        inputs[z-k] = iqn->states[iqn->index[z]];
                    }
                    ff_error_bp_model_multicore_opt(iqn->q_networks,iqn->q_network,1,1,iqn->state_size,inputs,min,min,next_q + k,NULL);
                    sum_models_partial_derivatives(iqn->q_network,iqn->q_networks,min);
                    for(z = k; z < k+min; z++){
                        reset_model_without_learning_parameters(iqn->q_networks[z-k]);
                    }
                }
            }
            
            // update model
            if(iqn->adaptive_clipping_flag){
                adaptive_gradient_clipping_model(iqn->q_network,iqn->adaptive_clipping_gradient_value,1e-3);
            }
            
            update_model(iqn->q_network,iqn->lr,iqn->momentum,iqn->batch_size,iqn->gd_flag,&iqn->beta1,&iqn->beta2,NO_REGULARIZATION,0,0,(unsigned long long int*)&iter);

            // update learning rate
            update_lr(&iqn->lr,iqn->lr_minimum,iqn->lr_maximum,iqn->initial_lr,iqn->lr_decay,iter,iqn->lr_epoch_threshold,iqn->lr_decay_flag);
            // update training parameters
            update_training_parameters(&iqn->beta1,&iqn->beta2,(unsigned long long int*)&iter,iqn->q_network->beta1_adam,iqn->q_network->beta2_adam);
            
            // reset the next_q values
            for(k = 0; k < iqn->batch_size; k++){
                set_vector_with_value(0,next_q[k],get_output_dimension_from_model(iqn->q_network));
            }
        }
        // save the network
        save_model(iqn->q_network, i);
    }
    
    free_matrix((void**)next_q, iqn->batch_size);
    free(inputs);
    free(models_to_handle);
}
