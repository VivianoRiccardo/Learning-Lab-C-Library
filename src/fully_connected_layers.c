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

/* This function builds a fully-connected layer according to the fcl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int input:= number of neurons of the previous layer
 *             @ int output:= number of neurons of the current layer
 *             @ int layer:= number of sequential layer [0,∞)
 *             @ int dropout_flag:= is set to 0 if you don't want to apply dropout, NO_DROPOUT (flag)
 *             @ int activation_flag:= is set to 0 if you don't want to apply the activation function else read in llab.h
 *             @ float dropout_threshold:= [0,1]
 *             @ int n_groups:= a number that divides the output in tot group for the layer normalization
 *             @ int normalization_flag:= either NO_NORMALIZATION or LAYER_NORMALIZATION 
 *             @ int training_mode:= either FREEZE_TRAINING or GRADIENT_DESCENT or EDGE_POPUP or FREEZE_BIASES [NOT COMPLETELY IMPLEMENTED THE LAST ONE]
 *             @ int feed_forward_flag:= either FULLY_FEED_FORWARD or EDGE_POPUP
 *                @ int mode:= STANDARD, NOISY 
 * */
fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode){
    if(input <= 0 || output <= 0 || layer < 0 || input*output <= 0){
        fprintf(stderr,"Error: input, output params must be > 0 and layer > -1\n");
        exit(1);
    }
    
    if (dropout_flag != NO_DROPOUT && dropout_flag != DROPOUT && dropout_flag != DROPOUT_TEST){
        fprintf(stderr,"Error, you must set the dropout flag properly!\n");
        exit(1);
    }
    
    if(activation_flag != NO_ACTIVATION && activation_flag != SIGMOID && activation_flag != TANH && activation_flag != RELU && activation_flag != SOFTMAX && activation_flag != LEAKY_RELU && activation_flag != ELU){
        fprintf(stderr,"Error, you must set the activation flag properly!\n");
        exit(1);
    } 
    
    if(dropout_threshold > 1 || dropout_threshold < 0){
        fprintf(stderr,"Error: you should set thedropout_threshold in [0,1]\n");
        exit(1);
    }
    
    if(training_mode != EDGE_POPUP && training_mode != FREEZE_TRAINING && training_mode != GRADIENT_DESCENT && training_mode != ONLY_FF){
        fprintf(stderr,"Error, you should set the training mode properly!\n");
        exit(1);
    }
    
    if(feed_forward_flag != FULLY_FEED_FORWARD && feed_forward_flag != EDGE_POPUP){
        fprintf(stderr,"Error: you should set your feed forward flag properly!\n");
        exit(1);
    }
    
    if(normalization_flag != GROUP_NORMALIZATION && normalization_flag != LAYER_NORMALIZATION && normalization_flag != LOCAL_RESPONSE_NORMALIZATION && normalization_flag != NO_NORMALIZATION){
        fprintf(stderr,"Error: you should set your normalization flag properly!\n");
        exit(1);
    }
    
    if(normalization_flag == GROUP_NORMALIZATION)
        normalization_flag = LAYER_NORMALIZATION;
    
    if(normalization_flag == LAYER_NORMALIZATION){
        if(n_groups <= 0 || output<=n_groups || output%n_groups != 0){
            fprintf(stderr,"Error: your groups must perfectly divide your output neurons\n");
            exit(1);
        }
    }
    
    if (normalization_flag != NO_NORMALIZATION && dropout_flag != NO_DROPOUT){
        fprintf(stderr,"Error: bad assignement dropout + normalization not recommended!\n");
        exit(1);
    }
    
    if((feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP) && normalization_flag == LAYER_NORMALIZATION){
        fprintf(stderr,"Error: layer normalization should not be matched with edge popup!\n");
        exit(1);
    }
    
    if(mode != STANDARD && mode != NOISY){
        fprintf(stderr,"Error: the fcl mode should be set to either standard or noisy!\n");
        exit(1);
    }
    
    int i,j;
    
    fcl* f = (fcl*)malloc(sizeof(fcl));
    f->input = input;
    f->output = output;
    f->layer = layer;
    f->mode = mode;
    f->dropout_flag = dropout_flag;
    f->activation_flag = activation_flag;
    f->dropout_threshold = dropout_threshold;
    f->training_mode = training_mode;
    f->feed_forward_flag = feed_forward_flag;
    f->normalization_flag = normalization_flag;
    if(f->feed_forward_flag != ONLY_DROPOUT){
        f->weights = (float*)malloc(sizeof(float)*output*input);
        if(mode == NOISY){
            f->noisy_weights = (float*)malloc(sizeof(float)*output*input);// new
            f->noisy_biases = (float*)calloc(output,sizeof(float));// new
        }
        else{
            f->noisy_weights = NULL;// new
            f->noisy_biases = NULL;// new
        }
        f->biases = (float*)calloc(output,sizeof(float));
        f->active_output_neurons = (int*)calloc(output,sizeof(int));
        if(f->training_mode != EDGE_POPUP && f->training_mode != ONLY_FF){
            f->d_weights = (float*)calloc(output*input,sizeof(float));
            f->d1_weights = (float*)calloc(output*input,sizeof(float));
            f->d2_weights = (float*)calloc(output*input,sizeof(float));
            f->d3_weights = (float*)calloc(output*input,sizeof(float));        
            f->d_biases = (float*)calloc(output,sizeof(float));
            f->d1_biases = (float*)calloc(output,sizeof(float));
            f->d2_biases = (float*)calloc(output,sizeof(float));
            f->d3_biases = (float*)calloc(output,sizeof(float));
            if(mode == NOISY){
                f->d_noisy_weights = (float*)calloc(output*input,sizeof(float));// new
                f->d1_noisy_weights = (float*)calloc(output*input,sizeof(float));// new
                f->d2_noisy_weights = (float*)calloc(output*input,sizeof(float));// new
                f->d3_noisy_weights = (float*)calloc(output*input,sizeof(float));// new
                f->d_noisy_biases = (float*)calloc(output,sizeof(float));// new
                f->d1_noisy_biases = (float*)calloc(output,sizeof(float));// new
                f->d2_noisy_biases = (float*)calloc(output,sizeof(float));// new
                f->d3_noisy_biases = (float*)calloc(output,sizeof(float));// new
            }
            else{
                f->d_noisy_weights = NULL;// new
                f->d1_noisy_weights = NULL;// new
                f->d2_noisy_weights = NULL;// new
                f->d3_noisy_weights = NULL;// new
                f->d_noisy_biases = NULL;// new
                f->d1_noisy_biases = NULL;// new
                f->d2_noisy_biases = NULL;// new
                f->d3_noisy_biases = NULL;// new
            }
        }
        
        else{
            f->d_weights = NULL;
            f->d1_weights = NULL;
            f->d2_weights = NULL;
            f->d3_weights = NULL;    
            f->d_noisy_weights = NULL;// new
            f->d1_noisy_weights = NULL;// new
            f->d2_noisy_weights = NULL;// new
            f->d3_noisy_weights = NULL;// new
            f->d_noisy_biases = NULL;// new
            f->d1_noisy_biases = NULL;// new
            f->d2_noisy_biases = NULL;// new
            f->d3_noisy_biases = NULL;// new
            f->d_biases = NULL;
            f->d1_biases = NULL;
            f->d2_biases = NULL;
            f->d3_biases = NULL;
        }
        f->pre_activation = (float*)calloc(output,sizeof(float));
    }
    
    else{
        f->weights = NULL;
        f->noisy_weights = NULL;// new
        f->noisy_biases = NULL;// new
        f->biases = NULL;
        f->active_output_neurons = NULL;
        f->d_weights = NULL;
        f->d1_weights = NULL;
        f->d2_weights = NULL;
        f->d3_weights = NULL;    
        f->d_noisy_weights = NULL;// new
        f->d1_noisy_weights = NULL;// new
        f->d2_noisy_weights = NULL;// new
        f->d3_noisy_weights = NULL;// new
        f->d_noisy_biases = NULL;// new
        f->d1_noisy_biases = NULL;// new
        f->d2_noisy_biases = NULL;// new
        f->d3_noisy_biases = NULL;// new
        f->d_biases = NULL;
        f->d1_biases = NULL;
        f->d2_biases = NULL;
        f->d3_biases = NULL;
        f->pre_activation = NULL;
    }
    if(dropout_flag != NO_DROPOUT){
        f->dropout_temp = (float*)calloc(output,sizeof(float));
        f->dropout_mask = (float*)calloc(output,sizeof(float));
    }
    else{
        f->dropout_temp = NULL;
        f->dropout_mask = NULL;
    }
    
    if(mode == NOISY){
        f->noise = (float*)calloc(input*output,sizeof(float));// new
        f->temp_weights = (float*)calloc(input*output,sizeof(float));// new
        f->noise_biases = (float*)calloc(output,sizeof(float));// new
        f->temp_biases = (float*)calloc(output,sizeof(float));// new
        set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
    }
    
    else{
        f->noise = NULL;
        f->temp_weights = NULL;
        f->noise_biases = NULL;
        f->temp_biases = NULL;
    }
        
    f->temp = (float*)calloc(output,sizeof(float));
    f->temp3 = (float*)calloc(output,sizeof(float));
    f->temp2 = (float*)calloc(input,sizeof(float));
    f->error2 = (float*)calloc(input,sizeof(float));
    
    if(f->feed_forward_flag != ONLY_DROPOUT && (f->feed_forward_flag == EDGE_POPUP || f->training_mode == EDGE_POPUP)){
        f->scores = (float*)calloc(output*input,sizeof(float));
        f->d_scores = (float*)calloc(output*input,sizeof(float));
        f->d1_scores = (float*)calloc(output*input,sizeof(float));
        f->d2_scores = (float*)calloc(output*input,sizeof(float));
        f->d3_scores = (float*)calloc(output*input,sizeof(float));
        f->indices = (int*)calloc(output*input,sizeof(int));
    }
    
    else{
        f->scores = NULL;
        f->d_scores  = NULL;
        f->d1_scores = NULL;
        f->d2_scores = NULL;
        f->d3_scores = NULL;
        f->indices = NULL;
    }
    if(f->activation_flag != NO_ACTIVATION && f->feed_forward_flag != ONLY_DROPOUT)
        f->post_activation = (float*)calloc(output,sizeof(float));
    else
        f->post_activation = NULL;
    if((f->normalization_flag == LAYER_NORMALIZATION || f->normalization_flag == LOCAL_RESPONSE_NORMALIZATION) && f->feed_forward_flag != ONLY_DROPOUT)
        f->post_normalization = (float*)calloc(output,sizeof(float));
    else
        f->post_normalization = NULL;
    f->k_percentage = 1;
    
    
    for(i = 0; i < output; i++){
        if(f->feed_forward_flag != ONLY_DROPOUT)
        f->active_output_neurons[i] = 1;
        for(j = 0; j < input; j++){
            if(f->feed_forward_flag != ONLY_DROPOUT){
                if(f->feed_forward_flag == EDGE_POPUP || f->training_mode == EDGE_POPUP){
                    f->indices[i*input+j] = i*input+j;
                    f->weights[i*input+j] = signed_kaiming_constant(input);
                    if(mode == NOISY){
                        f->noisy_weights[i*input+j] = 0.5/(sqrtf(2*input));// not from the paper, different init cause not xavier initialization but signed kaiming constant
                    }
                }
                else{
                    f->weights[i*input+j] = random_general_gaussian_xavier_init(input);
                    if(mode == NOISY){
                        f->noisy_weights[i*input+j] = 0.5/(sqrtf(input));// from the paper, in case of xavier initialization
                    }
                }
            }
        }
        if(dropout_flag)
            f->dropout_mask[i] = 1;
    }
    
    
    if(normalization_flag == LAYER_NORMALIZATION){
        f->layer_norm = batch_normalization(n_groups,output/n_groups);
    }
    else{
        f->layer_norm = NULL;
    }
    
    
    f->n_groups = n_groups;
    
    return f;
}
/* This function builds a fully-connected layer according to the fcl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int input:= number of neurons of the previous layer
 *             @ int output:= number of neurons of the current layer
 *             @ int layer:= number of sequential layer [0,∞)
 *             @ int dropout_flag:= is set to 0 if you don't want to apply dropout, NO_DROPOUT (flag)
 *             @ int activation_flag:= is set to 0 if you don't want to apply the activation function else read in llab.h
 *             @ float dropout_threshold:= [0,1]
 *             @ int n_groups:= a number that divides the output in tot group for the layer normalization
 *             @ int normalization_flag:= either NO_NORMALIZATION or LAYER_NORMALIZATION 
 *             @ int training_mode:= either FREEZE_TRAINING or GRADIENT_DESCENT or EDGE_POPUP or FREEZE_BIASES [NOT COMPLETELY IMPLEMENTED THE LAST ONE]
 *             @ int feed_forward_flag:= either FULLY_FEED_FORWARD or EDGE_POPUP
 * */
fcl* fully_connected_without_arrays(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode){
    if(input <= 0 || output <= 0 || layer < 0){
        return NULL;
    }
    
    if (dropout_flag != NO_DROPOUT && dropout_flag != DROPOUT && dropout_flag != DROPOUT_TEST){
        return NULL;
    }
    
    if(activation_flag != NO_ACTIVATION && activation_flag != SIGMOID && activation_flag != TANH && activation_flag != RELU && activation_flag != SOFTMAX && activation_flag != LEAKY_RELU && activation_flag != ELU){
        return NULL;
    } 
    
    if(dropout_threshold > 1 || dropout_threshold < 0){
        return NULL;
    }
    
    if(training_mode != EDGE_POPUP && training_mode != FREEZE_TRAINING && training_mode != GRADIENT_DESCENT){
        return NULL;
    }
    
    if(feed_forward_flag != FULLY_FEED_FORWARD && feed_forward_flag != EDGE_POPUP){
        return NULL;
    }
    
    if(normalization_flag != GROUP_NORMALIZATION && normalization_flag != LAYER_NORMALIZATION && normalization_flag != LOCAL_RESPONSE_NORMALIZATION && normalization_flag != NO_NORMALIZATION){
        return NULL;
    }
    
    if(normalization_flag == GROUP_NORMALIZATION)
        normalization_flag = LAYER_NORMALIZATION;
    
    if(normalization_flag == LAYER_NORMALIZATION){
        if(n_groups <= 0 || output<=n_groups || output%n_groups != 0){
            return NULL;
        }
    }
    
    if (normalization_flag != NO_NORMALIZATION && dropout_flag != NO_DROPOUT){
        return NULL;
    }
    
    if((feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP) && normalization_flag == LAYER_NORMALIZATION){
        return NULL;
    }
    
    if(mode != STANDARD && mode != NOISY){
        return NULL;
    }
 
    int i,j;
    
    fcl* f = (fcl*)malloc(sizeof(fcl));
    f->input = input;
    f->output = output;
    f->mode = mode;
    f->layer = layer;
    f->dropout_flag = dropout_flag;
    f->activation_flag = activation_flag;
    f->dropout_threshold = dropout_threshold;
    f->training_mode = training_mode;
    f->feed_forward_flag = feed_forward_flag;
    f->normalization_flag = normalization_flag;
    
    f->k_percentage = 1;
    
    
    
    
    
    if(normalization_flag == LAYER_NORMALIZATION){
        f->layer_norm = batch_normalization_without_arrays(n_groups,output/n_groups);
        if(f->layer_norm == NULL){
            free(f);
            return NULL;
        }
            
    }
    else{
        f->layer_norm = NULL;
    }
    
    
    f->n_groups = n_groups;
    
    return f;
}
/* This function builds a fully-connected layer according to the fcl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int input:= number of neurons of the previous layer
 *             @ int output:= number of neurons of the current layer
 *             @ int layer:= number of sequential layer [0,∞)
 *             @ int dropout_flag:= is set to 0 if you don't want to apply dropout, NO_DROPOUT (flag)
 *             @ int activation_flag:= is set to 0 if you don't want to apply the activation function else read in llab.h
 *             @ float dropout_threshold:= [0,1]
 *             @ int n_groups:= a number that divides the output in tot group for the layer normalization
 *             @ int normalization_flag:= either NO_NORMALIZATION or LAYER_NORMALIZATION 
 * */
fcl* fully_connected_without_learning_parameters(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, int mode){
    if(input <= 0 || output <= 0 || layer < 0){
        fprintf(stderr,"Error: input, output params must be > 0 and layer > -1\n");
        exit(1);
    }
    
    if (dropout_flag != NO_DROPOUT && dropout_flag != DROPOUT && dropout_flag != DROPOUT_TEST){
        fprintf(stderr,"Error, you must set the dropout flag properly!\n");
        exit(1);
    }
    
    if(activation_flag != NO_ACTIVATION && activation_flag != SIGMOID && activation_flag != TANH && activation_flag != RELU && activation_flag != SOFTMAX && activation_flag != LEAKY_RELU && activation_flag != ELU){
        fprintf(stderr,"Error, you must set the activation flag properly!\n");
        exit(1);
    } 
    
    if(dropout_threshold > 1 || dropout_threshold < 0){
        fprintf(stderr,"Error: you should set thedropout_threshold in [0,1]\n");
        exit(1);
    }
    
    if(training_mode != EDGE_POPUP && training_mode != FREEZE_TRAINING && training_mode != GRADIENT_DESCENT && training_mode != ONLY_FF){
        fprintf(stderr,"Error, you should set the training mode properly!\n");
        exit(1);
    }
    
    if(feed_forward_flag != FULLY_FEED_FORWARD && feed_forward_flag != EDGE_POPUP){
        fprintf(stderr,"Error: you should set your feed forward flag properly!\n");
        exit(1);
    }
    
    if(normalization_flag != GROUP_NORMALIZATION && normalization_flag != LAYER_NORMALIZATION && normalization_flag != LOCAL_RESPONSE_NORMALIZATION && normalization_flag != NO_NORMALIZATION){
        fprintf(stderr,"Error: you should set your normalization flag properly!\n");
        exit(1);
    }
    
    if(normalization_flag == GROUP_NORMALIZATION)
        normalization_flag = LAYER_NORMALIZATION;
    
    if(normalization_flag == LAYER_NORMALIZATION){
        if(n_groups <= 0 || output<=n_groups || output%n_groups != 0){
            fprintf(stderr,"Error: your groups must perfectly divide your output neurons\n");
            exit(1);
        }
    }
    
    if (normalization_flag != NO_NORMALIZATION && dropout_flag != NO_DROPOUT){
        fprintf(stderr,"Error: bad assignement dropout + normalization not recommended!\n");
        exit(1);
    }
    
    if((feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP) && normalization_flag == LAYER_NORMALIZATION){
        fprintf(stderr,"Error: layer normalization should not be matched with edge popup!\n");
        exit(1);
    }
    
    if(mode != NOISY && mode != STANDARD){
        fprintf(stderr,"Error: mode can be only either STANDARD or NOISY!\n");
        exit(1);
    }
 
    int i,j;
    
    fcl* f = (fcl*)malloc(sizeof(fcl));
    f->input = input;
    f->output = output;
    f->layer = layer;
    f->mode = mode;
    f->dropout_flag = dropout_flag;
    f->activation_flag = activation_flag;
    f->dropout_threshold = dropout_threshold;
    f->training_mode = training_mode;
    f->feed_forward_flag = feed_forward_flag;
    f->normalization_flag = normalization_flag;
    if(f->feed_forward_flag != ONLY_DROPOUT){
        f->weights = NULL;
        f->biases = NULL;
        f->noisy_weights = NULL;// new
        f->noisy_biases = NULL;
        f->active_output_neurons = NULL;
        if(f->training_mode != EDGE_POPUP && f->training_mode != ONLY_FF){
            f->d_weights = (float*)calloc(output*input,sizeof(float));
            f->d1_weights = NULL;
            f->d2_weights = NULL;
            f->d3_weights = NULL;
            f->d_biases = (float*)calloc(output,sizeof(float));
            f->d1_biases = NULL;
            f->d2_biases = NULL;
            f->d3_biases = NULL;
            if(mode == NOISY){
                f->d_noisy_weights = (float*)calloc(output*input,sizeof(float));// new
                f->d_noisy_biases = (float*)calloc(output,sizeof(float));// new
            } 
            
            else{
                f->d_noisy_weights = NULL;
                f->d_noisy_biases = NULL;
            }
            
            f->d1_noisy_weights = NULL;
            f->d2_noisy_weights = NULL;
            f->d3_noisy_weights = NULL;
            f->d1_noisy_biases = NULL;
            f->d2_noisy_biases = NULL;
            f->d3_noisy_biases = NULL;
                
            f->d1_biases = NULL;
            f->d2_biases = NULL;
            f->d3_biases = NULL;
            
            
        }
        
        else{
            f->d_weights = NULL;
            f->d1_weights = NULL;
            f->d2_weights = NULL;
            f->d3_weights = NULL;    
            f->d_noisy_weights = NULL;// new
            f->d1_noisy_weights = NULL;// new
            f->d2_noisy_weights = NULL;// new
            f->d3_noisy_weights = NULL;// new
            f->d_noisy_biases = NULL;// new
            f->d1_noisy_biases = NULL;// new
            f->d2_noisy_biases = NULL;// new
            f->d3_noisy_biases = NULL;// new
            f->d_biases = NULL;
            f->d1_biases = NULL;
            f->d2_biases = NULL;
            f->d3_biases = NULL;
        }
        f->pre_activation = (float*)calloc(output,sizeof(float));
    }
    
    else{
        f->weights = NULL;
        f->noisy_weights = NULL;// new
        f->noisy_biases = NULL;// new
        f->biases = NULL;
        f->active_output_neurons = NULL;
        f->d_weights = NULL;
        f->d1_weights = NULL;
        f->d2_weights = NULL;
        f->d3_weights = NULL;    
        f->d_noisy_weights = NULL;// new
        f->d1_noisy_weights = NULL;// new
        f->d2_noisy_weights = NULL;// new
        f->d3_noisy_weights = NULL;// new
        f->d_noisy_biases = NULL;// new
        f->d1_noisy_biases = NULL;// new
        f->d2_noisy_biases = NULL;// new
        f->d3_noisy_biases = NULL;// new
        f->d_biases = NULL;
        f->d1_biases = NULL;
        f->d2_biases = NULL;
        f->d3_biases = NULL;
        f->pre_activation = NULL;
    }
    if(dropout_flag != NO_DROPOUT){
        f->dropout_temp = (float*)calloc(output,sizeof(float));
        f->dropout_mask = (float*)calloc(output,sizeof(float));
    }
    else{
        f->dropout_temp = NULL;
        f->dropout_mask = NULL;
    }
    
    if(mode == NOISY){
        f->noise = (float*)calloc(input*output,sizeof(float));// new
        f->temp_weights = (float*)calloc(input*output,sizeof(float));// new
        f->noise_biases = (float*)calloc(output,sizeof(float));// new
        f->temp_biases = (float*)calloc(output,sizeof(float));// new
        set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
    }
    
    else{
        f->noise = NULL;// new
        f->temp_weights = NULL;// new
        f->noise_biases = NULL;// new
        f->temp_biases = NULL;// new
    }
    
    f->temp = (float*)calloc(output,sizeof(float));
    f->temp3 = (float*)calloc(output,sizeof(float));
    f->temp2 = (float*)calloc(input,sizeof(float));
    f->error2 = (float*)calloc(input,sizeof(float));
    
    if(f->feed_forward_flag != ONLY_DROPOUT && (f->feed_forward_flag == EDGE_POPUP || f->training_mode == EDGE_POPUP)){
        f->scores = NULL;
        f->d_scores = (float*)calloc(output*input,sizeof(float));
        f->d1_scores = NULL;
        f->d2_scores = NULL;
        f->d3_scores = NULL;
        f->indices = NULL;
    }
    
    else{
        f->scores = NULL;
        f->d_scores  = NULL;
        f->d1_scores = NULL;
        f->d2_scores = NULL;
        f->d3_scores = NULL;
        f->indices = NULL;
    }
    if(f->activation_flag != NO_ACTIVATION && f->feed_forward_flag != ONLY_DROPOUT)
        f->post_activation = (float*)calloc(output,sizeof(float));
    else
        f->post_activation = NULL;
    if((f->normalization_flag == LAYER_NORMALIZATION || f->normalization_flag == LOCAL_RESPONSE_NORMALIZATION) && f->feed_forward_flag != ONLY_DROPOUT)
        f->post_normalization = (float*)calloc(output,sizeof(float));
    else
        f->post_normalization = NULL;
    f->k_percentage = 1;
    
    
    for(i = 0; i < output; i++){
        if(dropout_flag)
            f->dropout_mask[i] = 1;
    }
    
    
    if(normalization_flag == LAYER_NORMALIZATION){
        f->layer_norm = batch_normalization_without_learning_parameters(n_groups,output/n_groups);
    }
    else{
        f->layer_norm = NULL;
    }
    
    
    f->n_groups = n_groups;
    
    return f;
}

int is_noisy(fcl* f){// new
    return f->mode == NOISY;
}

int exists_params_fcl(fcl* f){
    return f->feed_forward_flag != ONLY_DROPOUT;
}

int exists_d_params_fcl(fcl* f){
    return f->feed_forward_flag != ONLY_DROPOUT && f->training_mode != EDGE_POPUP && f->training_mode != ONLY_FF;
}

int exists_dropout_stuff_fcl(fcl* f){
    return f->dropout_flag != NO_DROPOUT;
}

int exists_edge_popup_stuff_fcl(fcl* f){
    return f->feed_forward_flag != ONLY_DROPOUT && (f->feed_forward_flag == EDGE_POPUP || f->training_mode == EDGE_POPUP);
}

int exists_activation_fcl(fcl* f){
    return f->activation_flag != NO_ACTIVATION && f->feed_forward_flag != ONLY_DROPOUT;
}

int exists_normalization_fcl(fcl* f){
    return (f->normalization_flag == LAYER_NORMALIZATION || f->normalization_flag == LOCAL_RESPONSE_NORMALIZATION) && f->feed_forward_flag != ONLY_DROPOUT;
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->weights);
    free(f->d_weights);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->d3_weights);
    free(f->noisy_weights);// new
    free(f->d_noisy_weights);// new
    free(f->d1_noisy_weights);// new
    free(f->d2_noisy_weights);// new
    free(f->d3_noisy_weights);// new
    free(f->noisy_biases);// new
    free(f->d_noisy_biases);// new
    free(f->d1_noisy_biases);// new
    free(f->d2_noisy_biases);// new
    free(f->d3_noisy_biases);// new
    free(f->biases);
    free(f->d_biases);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->d3_biases);
    free(f->pre_activation);
    free(f->post_activation);
    free(f->post_normalization);
    free(f->dropout_mask);
    free(f->dropout_temp);
    free(f->noise);// new
    free(f->temp_weights);// new
    free(f->noise_biases);// new
    free(f->temp_biases);// new
    free(f->temp);
    free(f->temp2);
    free(f->temp3);
    free(f->error2);
    free(f->scores);
    free(f->d_scores);
    free(f->d1_scores);
    free(f->d2_scores);
    free(f->d3_scores);
    free(f->indices);
    free(f->active_output_neurons);
    free_batch_normalization(f->layer_norm);
    free(f);    
}
/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected_without_arrays(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->layer_norm);
    free(f);    
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected_for_edge_popup(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->d_weights);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->d3_weights);
    free(f->d_noisy_weights);// new
    free(f->d1_noisy_weights);// new
    free(f->d2_noisy_weights);// new
    free(f->d3_noisy_weights);// new
    free(f->d_noisy_biases);// new
    free(f->d1_noisy_biases);// new
    free(f->d2_noisy_biases);// new
    free(f->d3_noisy_biases);// new
    free(f->d_biases);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->d3_biases);
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected_complementary_edge_popup(fcl* f){
    if(f == NULL){
        return;
    }
    free(f->weights);
    free(f->noisy_weights);// new
    free(f->noisy_biases);// new
    free(f->biases);
    free(f->pre_activation);
    free(f->post_activation);
    free(f->post_normalization);
    free(f->dropout_mask);
    free(f->dropout_temp);
    free(f->noise);// new
    free(f->temp_weights);// new
    free(f->noise_biases);// new
    free(f->temp_biases);// new
    free(f->temp);
    free(f->temp2);
    free(f->temp3);
    free(f->error2);
    free(f->scores);
    free(f->d_scores);
    free(f->d1_scores);
    free(f->d2_scores);
    free(f->d3_scores);
    free(f->indices);
    free(f->active_output_neurons);
    free_batch_normalization(f->layer_norm);
    free(f);
}
/* This function saves a fully-connected layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ fcl* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_fcl(fcl* f, int n){
    if(f == NULL || n<0)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    
    convert_data(&f->mode,sizeof(int),1);// new
    i = fwrite(&f->mode,sizeof(int),1,fw);// new
    convert_data(&f->mode,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");// new
        exit(1);// new
    }
    
    convert_data(&f->n_groups,sizeof(int),1);
    i = fwrite(&f->n_groups,sizeof(int),1,fw);
    convert_data(&f->n_groups,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->normalization_flag,sizeof(int),1);
    i = fwrite(&f->normalization_flag,sizeof(int),1,fw);
    convert_data(&f->normalization_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->feed_forward_flag,sizeof(int),1);
    i = fwrite(&f->feed_forward_flag,sizeof(int),1,fw);
    convert_data(&f->feed_forward_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->training_mode,sizeof(int),1);
    i = fwrite(&f->training_mode,sizeof(int),1,fw);
    convert_data(&f->training_mode,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->input,sizeof(int),1);
    i = fwrite(&f->input,sizeof(int),1,fw);
    convert_data(&f->input,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->output,sizeof(int),1);
    i = fwrite(&f->output,sizeof(int),1,fw);
    convert_data(&f->output,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->layer,sizeof(int),1);
    i = fwrite(&f->layer,sizeof(int),1,fw);
    convert_data(&f->layer,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->dropout_flag,sizeof(int),1);
    i = fwrite(&f->dropout_flag,sizeof(int),1,fw);
    convert_data(&f->dropout_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->activation_flag,sizeof(int),1);
    i = fwrite(&f->activation_flag,sizeof(int),1,fw);
    convert_data(&f->activation_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    convert_data(&f->dropout_threshold,sizeof(float),1);
    i = fwrite(&f->dropout_threshold,sizeof(float),1,fw);
    convert_data(&f->dropout_threshold,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    if(exists_params_fcl(f)){
        convert_data(f->weights,sizeof(float),(f->input)*(f->output));
        i = fwrite(f->weights,sizeof(float)*(f->input)*(f->output),1,fw);
        convert_data(f->weights,sizeof(float),(f->input)*(f->output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
            exit(1);
        }
        convert_data(f->biases,sizeof(float),(f->output));
        i = fwrite(f->biases,sizeof(float)*(f->output),1,fw);
        convert_data(f->biases,sizeof(float),(f->output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
            exit(1);
        }
        convert_data(f->active_output_neurons,sizeof(int),(f->output));
        i = fwrite(f->active_output_neurons,sizeof(int)*(f->output),1,fw);
        convert_data(f->active_output_neurons,sizeof(int),(f->output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
            exit(1);
        }
    }
    
    if(is_noisy(f)){// new
        convert_data(f->noisy_weights,sizeof(float),(f->input)*(f->output));// new
        i = fwrite(f->noisy_weights,sizeof(float)*(f->input)*(f->output),1,fw);// new
        convert_data(f->noisy_weights,sizeof(float),(f->input)*(f->output));// new
        if(i != 1){// new
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");// new
            exit(1);// new
        }// new
        convert_data(f->noisy_biases,sizeof(float),(f->output));// new
        i = fwrite(f->noisy_biases,sizeof(float)*(f->output),1,fw);// new
        convert_data(f->noisy_biases,sizeof(float),(f->output));// new
        if(i != 1){// new
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");// new
            exit(1);// new
        }// new
    }
    
    if(exists_edge_popup_stuff_fcl(f)){
        convert_data(f->scores,sizeof(float),(f->output)*(f->input));
        i = fwrite(f->scores,sizeof(float)*(f->output)*(f->input),1,fw);
        convert_data(f->scores,sizeof(float),(f->output)*(f->input));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
            exit(1);
        }
        convert_data(f->indices,sizeof(int),(f->output)*(f->input));
        i = fwrite(f->indices,sizeof(int)*(f->output)*(f->input),1,fw);
        convert_data(f->indices,sizeof(int),(f->output)*(f->input));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
            exit(1);
        }
    }
    
    
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        save_bn(f->layer_norm,n);
    
    free(s);
    
}


/* This function copies the values in weights and biases vector in the weights 
 * and biases vector of a fcl structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the structure
 *             @ float* weights:= the weights that must be copied (size = f->output*f->input)
 *             @ float* biases:= the biases that must be copied (size = f->output)
 * 
 * */
void copy_fcl_params(fcl* f, float* weights,float* noisy_weights, float* noisy_biases, float* biases){//new
    if(exists_params_fcl(f)){
        copy_array(weights,f->weights,f->input*f->output);
        copy_array(biases,f->biases,f->output);
    }
    if(is_noisy(f)){// new
        copy_array(noisy_weights,f->noisy_weights,f->input*f->output);// new
        copy_array(noisy_biases,f->noisy_biases,f->output);// new
    }// new
}


/* This function loads a fully-connected layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
fcl* load_fcl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int input = 0,output = 0,layer = 0,dropout_flag = 0,activation_flag = 0, training_mode = 0,feed_forward_flag = 0, n_groups = 0, normalization_flag = 0;
    int mode = STANDARD;// new
    float dropout_threshold = 0;
    float* weights = NULL;
    float* noisy_weights = NULL;// new
    float* noisy_biases = NULL;// new
    float* biases = NULL;
    float* scores = NULL;
    int* indices = NULL;
    int* active_output_neurons = NULL;
    bn* layer_norm = NULL;
    
    i = fread(&mode,sizeof(int),1,fr);
    convert_data(&mode,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&n_groups,sizeof(int),1,fr);
    convert_data(&n_groups,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&normalization_flag,sizeof(int),1,fr);
    convert_data(&normalization_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&feed_forward_flag,sizeof(int),1,fr);
    convert_data(&feed_forward_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&training_mode,sizeof(int),1,fr);
    convert_data(&training_mode,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&input,sizeof(int),1,fr);
    convert_data(&input,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&output,sizeof(int),1,fr);
    convert_data(&output,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&layer,sizeof(int),1,fr);
    convert_data(&layer,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&dropout_flag,sizeof(int),1,fr);
    convert_data(&dropout_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    convert_data(&activation_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&dropout_threshold,sizeof(float),1,fr);
    convert_data(&dropout_threshold,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    if(feed_forward_flag != ONLY_DROPOUT){
        weights = (float*)malloc(sizeof(float)*input*output);
        if(mode == NOISY){// new
            noisy_weights = (float*)malloc(sizeof(float)*input*output);// new
            noisy_biases = (float*)malloc(sizeof(float)*output);// new
        }// new
        active_output_neurons = (int*)malloc(sizeof(int)*output);
        biases = (float*)malloc(sizeof(float)*output);
    }
    
    if(feed_forward_flag != ONLY_DROPOUT && (feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP)){
        scores = (float*)malloc(sizeof(float)*input*output);
        indices = (int*)malloc(sizeof(int)*input*output);
    }    
    
    if(feed_forward_flag != ONLY_DROPOUT){
        i = fread(weights,sizeof(float)*(input)*(output),1,fr);
        convert_data(weights,sizeof(float),(input)*(output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
            exit(1);
        }
        
        i = fread(biases,sizeof(float)*(output),1,fr);
        convert_data(biases,sizeof(float),(output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
            exit(1);
        }
        
        i = fread(active_output_neurons,sizeof(int)*(output),1,fr);
        convert_data(active_output_neurons,sizeof(int),(output));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
            exit(1);
        }
    }
    
    if(mode == NOISY){// new
        i = fread(noisy_weights,sizeof(float)*(input)*(output),1,fr);// new
        convert_data(noisy_weights,sizeof(float),(input)*(output));// new
        if(i != 1){// new
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");// new
            exit(1);// new
        }// new
        i = fread(noisy_biases,sizeof(float)*(output),1,fr);// new
        convert_data(noisy_biases,sizeof(float),(output));// new
        if(i != 1){// new
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");// new
            exit(1);// new
        }// new
    }// new
    
    if(feed_forward_flag != ONLY_DROPOUT && (feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP)){
        i = fread(scores,sizeof(float)*(output)*(input),1,fr);
        convert_data(scores,sizeof(float),(output)*(input));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
            exit(1);
        }
        
        i = fread(indices,sizeof(int)*(output)*(input),1,fr);
        convert_data(indices,sizeof(int),(output)*(input));
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
            exit(1);
        }
    }
    
    
    
    fcl* f = fully_connected(input,output,layer,dropout_flag,activation_flag,dropout_threshold, n_groups, normalization_flag,training_mode,feed_forward_flag,mode);// new
    copy_fcl_params(f,weights,noisy_weights,noisy_biases, biases);// new
    if(exists_edge_popup_stuff_fcl(f)){
        copy_array(scores,f->scores,input*output);
        copy_int_array(indices,f->indices,input*output);
    }
    if(exists_params_fcl(f))
        copy_int_array(active_output_neurons,f->active_output_neurons,output);
    free(weights);
    free(noisy_weights);// new
    free(noisy_biases);// new
    free(biases);
    free(indices);
    free(active_output_neurons);
    free(scores);
    
    if(normalization_flag == LAYER_NORMALIZATION){
        layer_norm = load_bn(fr);
        paste_bn(layer_norm,f->layer_norm);
    }
    free_batch_normalization(layer_norm);
    return f;
}


/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array, and all the arrays used by ff and bp.
 * You have a fcl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 * 
 * */
fcl* copy_fcl(fcl* f){
    if(f == NULL)
        return NULL;
    fcl* copy = fully_connected(f->input, f->output,f->layer, f->dropout_flag,f->activation_flag,f->dropout_threshold,f->n_groups,f->normalization_flag, f->training_mode,f->feed_forward_flag, f->mode);// new
    if(exists_params_fcl(f)){
        copy_array(f->weights,copy->weights,f->output*f->input);
        copy_array(f->biases,copy->biases,f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    }
    if(is_noisy(f)){// new
        copy_array(f->noisy_weights,copy->noisy_weights,f->output*f->input);// new
        copy_array(f->noisy_biases,copy->noisy_biases,f->output);// new
    }// new
    if(exists_d_params_fcl(f)){
        if(f->d_weights != NULL)
        copy_array(f->d_weights,copy->d_weights,f->output*f->input);
        if(f->d1_weights != NULL)
        copy_array(f->d1_weights,copy->d1_weights,f->output*f->input);
        if(f->d2_weights != NULL)
        copy_array(f->d2_weights,copy->d2_weights,f->output*f->input);
        if(f->d3_weights != NULL)
        copy_array(f->d3_weights,copy->d3_weights,f->output*f->input);
        if(is_noisy(f)){// new
            if(f->d_noisy_weights != NULL)
            copy_array(f->d_noisy_weights,copy->d_noisy_weights,f->output*f->input);// new
            if(f->d1_noisy_weights != NULL)
            copy_array(f->d1_noisy_weights,copy->d1_noisy_weights,f->output*f->input);// new
            if(f->d2_noisy_weights != NULL)
            copy_array(f->d2_noisy_weights,copy->d2_noisy_weights,f->output*f->input);// new
            if(f->d3_noisy_weights != NULL)
            copy_array(f->d3_noisy_weights,copy->d3_noisy_weights,f->output*f->input);// new
            if(f->d_noisy_biases != NULL)
            copy_array(f->d_noisy_biases,copy->d_noisy_biases,f->output);// new
            if(f->d1_noisy_biases != NULL)
            copy_array(f->d1_noisy_biases,copy->d1_noisy_biases,f->output);// new
            if(f->d2_noisy_biases != NULL)
            copy_array(f->d2_noisy_biases,copy->d2_noisy_biases,f->output);// new
            if(f->d3_noisy_biases != NULL)
            copy_array(f->d3_noisy_biases,copy->d3_noisy_biases,f->output);// new
        }// new
        if(f->d_biases != NULL)
        copy_array(f->d_biases,copy->d_biases,f->output);
        if(f->d1_biases != NULL)
        copy_array(f->d1_biases,copy->d1_biases,f->output);
        if(f->d2_biases != NULL)
        copy_array(f->d2_biases,copy->d2_biases,f->output);
        if(f->d3_biases != NULL)
        copy_array(f->d3_biases,copy->d3_biases,f->output);
    }
    
    if(exists_edge_popup_stuff_fcl(f)){
        if(f->scores != NULL)
        copy_array(f->scores,copy->scores,f->input*f->output);
        if(f->d_scores != NULL)
        copy_array(f->d_scores,copy->d_scores,f->input*f->output);
        if(f->d1_scores != NULL)
        copy_array(f->d1_scores,copy->d1_scores,f->input*f->output);
        if(f->d2_scores != NULL)
        copy_array(f->d2_scores,copy->d2_scores,f->input*f->output);
        if(f->d3_scores != NULL)
        copy_array(f->d3_scores,copy->d3_scores,f->input*f->output);
        if(f->indices != NULL)
        copy_int_array(f->indices,copy->indices,f->input*f->output);
    }
    if(f->normalization_flag == LAYER_NORMALIZATION){
        paste_bn(f->layer_norm,copy->layer_norm);
    }
    copy->k_percentage = f->k_percentage;
    return copy;
}
/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array, and all the arrays used by ff and bp.
 * You have a fcl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 * 
 * */
fcl* copy_fcl_without_learning_parameters(fcl* f){
    if(f == NULL)
        return NULL;
    fcl* copy = fully_connected_without_learning_parameters(f->input, f->output,f->layer, f->dropout_flag,f->activation_flag,f->dropout_threshold,f->n_groups,f->normalization_flag, f->training_mode,f->feed_forward_flag, f->mode);// new
    if(exists_d_params_fcl(f)){
        copy_array(f->d_weights,copy->d_weights,f->output*f->input);
        copy_array(f->d_biases,copy->d_biases,f->output);
        if(is_noisy(f)){// new
            copy_array(f->d_noisy_weights,copy->d_noisy_weights,f->output*f->input);// new
            copy_array(f->d_noisy_biases,copy->d_noisy_biases,f->output);// new
        }// new
    }
    
    if(exists_edge_popup_stuff_fcl(f)){
        copy_array(f->d_scores,copy->d_scores,f->input*f->output);
    }
    if(f->normalization_flag == LAYER_NORMALIZATION){
        paste_bn_without_learning_parameters(f->layer_norm,copy->layer_norm);
    }
    copy->k_percentage = f->k_percentage;
    return copy;
}


/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array, and all the arrays used by ff and bp.
 * You have a fcl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 * 
 * */
fcl* copy_light_fcl(fcl* f){
    fcl* copy = copy_fcl(f);
    free_fully_connected_for_edge_popup(copy);
    return copy;
}

/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f)){
    set_vector_with_value(0,f->pre_activation,f->output);}
    if(exists_activation_fcl(f)){
    set_vector_with_value(0,f->post_activation,f->output);}
    if(exists_normalization_fcl(f)){
    set_vector_with_value(0,f->post_normalization,f->output);}
    if(exists_d_params_fcl(f)){
    set_vector_with_value(0,f->d_biases,f->output);}
    if(exists_dropout_stuff_fcl(f)){
        if(f->dropout_mask != NULL){
            set_vector_with_value(1,f->dropout_mask,f->output);
        }
        if(f->dropout_temp != NULL){
            set_vector_with_value(1,f->dropout_temp,f->output);
        }
    }
    if(is_noisy(f)){// new
        if(f->noise != NULL && f->noise_biases != NULL)
            set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    set_vector_with_value(0,f->temp,f->output);
    set_vector_with_value(0,f->temp3,f->output);
    set_vector_with_value(0,f->temp2,f->input);
    set_vector_with_value(0,f->error2,f->input);
    if(exists_d_params_fcl(f)){
        set_vector_with_value(0,f->d_weights,size);
        if(is_noisy(f)){// new
            if(f->d_noisy_weights != NULL && f->d_noisy_biases != NULL)
                set_vector_with_value(0,f->d_noisy_weights,size);// new
                set_vector_with_value(0,f->d_noisy_biases,f->output);// new
        }// new
    }
    if(f->training_mode == EDGE_POPUP){
        for(i = 0; i < f->output*f->input; i++){
                f->indices[i] = i;
                f->d_scores[i] = 0;
        }
        sort(f->scores,f->indices,0,f->output*f->input-1);
        free(f->active_output_neurons);
        f->active_output_neurons = get_used_outputs(f,NULL,FCLS,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_only_for_ff(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f)){
    set_vector_with_value(0,f->pre_activation,f->output);}
    if(exists_activation_fcl(f)){
    set_vector_with_value(0,f->post_activation,f->output);}
    if(exists_normalization_fcl(f)){
    set_vector_with_value(0,f->post_normalization,f->output);}
    if(exists_dropout_stuff_fcl(f)){
        if(f->dropout_mask != NULL){
            set_vector_with_value(1,f->dropout_mask,f->output);
        }
        if(f->dropout_temp != NULL){
            set_vector_with_value(1,f->dropout_temp,f->output);
        }
    }
    if(is_noisy(f)){// new
        if(f->noise != NULL && f->noise_biases != NULL)
            set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}

/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_without_learning_parameters(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f))
    set_vector_with_value(0,f->pre_activation,f->output);
    if(exists_activation_fcl(f))
    set_vector_with_value(0,f->post_activation,f->output);
    if(exists_normalization_fcl(f))
    set_vector_with_value(0,f->post_normalization,f->output);
    if(exists_d_params_fcl(f))
    set_vector_with_value(0,f->d_biases,f->output);
    if(exists_dropout_stuff_fcl(f)){
        if(f->dropout_mask != NULL)
            set_vector_with_value(1,f->dropout_mask,f->output);
        if(f->dropout_temp != NULL)
            set_vector_with_value(1,f->dropout_temp,f->output);
    }
    if(is_noisy(f)){// new
        if(f->noise != NULL && f->noise_biases != NULL)
            set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    set_vector_with_value(0,f->temp,f->output);
    set_vector_with_value(0,f->temp3,f->output);
    set_vector_with_value(0,f->temp2,f->input);
    set_vector_with_value(0,f->error2,f->input);
    if(exists_d_params_fcl(f)){
        set_vector_with_value(0,f->d_weights,size);
        if(is_noisy(f)){// new
            set_vector_with_value(0,f->d_noisy_weights,size);// new
            set_vector_with_value(0,f->d_noisy_biases,f->output);// new
        }// new
    }
    
    if(f->training_mode == EDGE_POPUP)
        set_vector_with_value(0,f->d_scores,size);
    
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change [DEPRECATED]
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_except_partial_derivatives(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f)){
    set_vector_with_value(0,f->pre_activation,f->output);}
    if(exists_activation_fcl(f)){
    set_vector_with_value(0,f->post_activation,f->output);}
    if(exists_normalization_fcl(f)){
    set_vector_with_value(0,f->post_normalization,f->output);}
    if(exists_dropout_stuff_fcl(f)){
        if(f->dropout_mask != NULL){
            set_vector_with_value(1,f->dropout_mask,f->output);
        }
        if(f->dropout_temp != NULL){
            set_vector_with_value(1,f->dropout_temp,f->output);
        }
    }
    if(is_noisy(f)){// new
        if(f->noise != NULL && f->noise_biases != NULL)
            set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    set_vector_with_value(0,f->temp,f->output);
    set_vector_with_value(0,f->temp3,f->output);
    set_vector_with_value(0,f->temp2,f->input);
    set_vector_with_value(0,f->error2,f->input);

    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn_except_partial_derivatives(f->layer_norm);
    return f;
}

/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_without_dwdb(fcl* f){
   if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f))
    set_vector_with_value(0,f->pre_activation,f->output);
    if(exists_activation_fcl(f))
    set_vector_with_value(0,f->post_activation,f->output);
    if(exists_normalization_fcl(f))
    set_vector_with_value(0,f->post_normalization,f->output);
    if(exists_dropout_stuff_fcl(f)){
        set_vector_with_value(1,f->dropout_mask,f->output);
        set_vector_with_value(1,f->dropout_temp,f->output);
    }
    if(is_noisy(f)){// new
        set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    set_vector_with_value(0,f->temp,f->output);
    set_vector_with_value(0,f->temp3,f->output);
    set_vector_with_value(0,f->temp2,f->input);
    set_vector_with_value(0,f->error2,f->input);
    
    if(f->training_mode == EDGE_POPUP){
        for(i = 0; i < f->output*f->input; i++){
                f->indices[i] = i;
                f->d_scores[i] = 0;
        }
        sort(f->scores,f->indices,0,f->output*f->input-1);
        free(f->active_output_neurons);
        f->active_output_neurons = get_used_outputs(f,NULL,FCLS,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_without_dwdb_without_learning_parameters(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    if(exists_params_fcl(f))
    set_vector_with_value(0,f->pre_activation,f->output);
    if(exists_activation_fcl(f))
    set_vector_with_value(0,f->post_activation,f->output);
    if(exists_normalization_fcl(f))
    set_vector_with_value(0,f->post_normalization,f->output);
    if(exists_dropout_stuff_fcl(f)){
        set_vector_with_value(1,f->dropout_mask,f->output);
        set_vector_with_value(1,f->dropout_temp,f->output);
    }
    if(is_noisy(f)){// new
        set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    set_vector_with_value(0,f->temp,f->output);
    set_vector_with_value(0,f->temp3,f->output);
    set_vector_with_value(0,f->temp2,f->input);
    set_vector_with_value(0,f->error2,f->input);
    
    
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl_for_edge_popup(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    int size = f->input*f->output;
    
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output){
            if(exists_params_fcl(f))
            f->pre_activation[i] = 0;
            if(exists_activation_fcl(f))
            f->post_activation[i] = 0;
            if(exists_normalization_fcl(f))
            f->post_normalization[i] = 0;
            if(exists_dropout_stuff_fcl(f)){
                f->dropout_mask[i] = 1;
                f->dropout_temp[i] = 0;
            }
            f->temp[i] = 0;
            f->temp3[i] = 0;
            
        }
        if(i < f->input){
            f->temp2[i] = 0;
            f->error2[i] = 0;
        }
        
        if(f->training_mode == EDGE_POPUP){
            f->d_scores[i] = 0;
        }

    }
    if(is_noisy(f)){// new
        set_factorised_noise(f->input, f->output, f->noise, f->noise_biases);
        set_vector_with_value(0,f->temp_weights,size);// new
        set_vector_with_value(0,f->temp_biases,f->output);// new
    }// new
    if(exists_d_params_fcl(f)){
        set_vector_with_value(0,f->d_weights,size);
        if(is_noisy(f)){// new
            set_vector_with_value(0,f->d_noisy_weights,size);// new
            set_vector_with_value(0,f->d_noisy_biases,f->output);// new
        }// new
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn_except_partial_derivatives(f->layer_norm);
    return f;
}

void free_scores(fcl* f){
    free(f->scores);
    f->scores = NULL;
}

void free_indices(fcl* f){
    free(f->indices);
    f->indices = NULL;
}

void set_null_scores(fcl* f){
    f->scores = NULL;
}

void set_null_indices(fcl* f){
    f->indices = NULL;
}


/* this function returns the space allocated by the arrays of f (more or less)
 * 
 * Input:
 * 
 *             fcl* f:= the fully-connected layer f
 * 
 * */
uint64_t size_of_fcls(fcl* f){
    if(f == NULL)
        return 0;
    uint64_t sum = 0;
    if(exists_params_fcl(f)){
        sum+=(f->output*2+f->output*f->input)*sizeof(float) + f->output*sizeof(int);
        if(is_noisy(f)){// new
            sum+=f->output*f->input*sizeof(float)*4 + f->output*sizeof(float)*4;// new
        }// new
    }
    
    if(exists_d_params_fcl(f)){
        sum+=(f->output*f->input + f->output)*4*sizeof(float);
        if(is_noisy(f)){// new
            sum+=(f->output*f->input)*4*sizeof(float);// n ew
        }// new
    }
    if(exists_edge_popup_stuff_fcl(f)){
        sum+=f->output*f->input*5*sizeof(float) +f->output*f->input*sizeof(int);
    }
    if(exists_dropout_stuff_fcl(f)){
        sum+=f->output*2*sizeof(float);
    }
    
    if(exists_activation_fcl(f)){
        sum+=f->output*sizeof(float);
    }
    if(exists_normalization_fcl(f)){
        sum+=f->output*sizeof(float);
        sum+=size_of_bn(f->layer_norm);
    }
    
    sum+=(f->output+f->input)*2*sizeof(float);
    if(is_noisy(f)){// new
        sum+=(f->output*f->input*2*sizeof(float)) + f->output*2*sizeof(float);// new
    }// new
    return sum;
}
/* this function returns the space allocated by the arrays of f (more or less)
 * 
 * Input:
 * 
 *             fcl* f:= the fully-connected layer f
 * 
 * */
uint64_t size_of_fcls_without_learning_parameters(fcl* f){
    if(f==NULL)
        return 0;
    uint64_t sum = 0;

    if(exists_d_params_fcl(f)){
        sum+=(f->output*f->input + f->output)*sizeof(float);
        if(is_noisy(f)){// new
            sum+=f->output*f->input*sizeof(float) + f->output*sizeof(float);// n ew
        }// new
    }
    if(exists_edge_popup_stuff_fcl(f)){
        sum+=f->output*f->input*5*sizeof(float);
    }
    if(exists_dropout_stuff_fcl(f)){
        sum+=f->output*2*sizeof(float);
    }
    
    if(exists_activation_fcl(f)){
        sum+=f->output*sizeof(float);
    }
    if(exists_normalization_fcl(f)){
        sum+=f->output*sizeof(float);
        sum+=size_of_bn_without_learning_parameters(f->layer_norm);
    }
    
    sum+=(f->output+f->input)*2*sizeof(float);
    if(is_noisy(f)){// new
        sum+=(f->output*f->input*2*sizeof(float)) + f->output*2*sizeof(float);// new
    }// new
    return sum;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_fcl(fcl* f,fcl* copy){
    if(f == NULL || copy == NULL)
        return;
    if(f->output != copy->output)
        return;
    int in_out = f->output*f->input;
    if(in_out != copy->input*copy->output)
        return;
    copy->k_percentage = f->k_percentage;
    if(exists_params_fcl(f) && exists_params_fcl(copy)){
        copy_array(f->weights,copy->weights,in_out);
        copy_array(f->biases,copy->biases,f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
        if(is_noisy(f) && is_noisy(copy)){// new
            copy_array(f->noisy_weights,copy->noisy_weights,in_out);// new
            copy_array(f->noisy_biases,copy->noisy_biases,f->output);// new
        }// new
    }
    if(exists_d_params_fcl(f) && exists_d_params_fcl(copy)){
        if(f->d_weights != NULL)
        copy_array(f->d_weights,copy->d_weights,in_out);
        if(f->d1_weights != NULL)
        copy_array(f->d1_weights,copy->d1_weights,in_out);
        if(f->d2_weights != NULL)
        copy_array(f->d2_weights,copy->d2_weights,in_out);
        if(f->d3_weights != NULL)
        copy_array(f->d3_weights,copy->d3_weights,in_out);
        if(is_noisy(f) && is_noisy(copy)){// new
            if(f->d_noisy_weights != NULL)
            copy_array(f->d_noisy_weights,copy->d_noisy_weights,in_out);// new
            if(f->d1_noisy_weights != NULL)
            copy_array(f->d1_noisy_weights,copy->d1_noisy_weights,in_out);// new
            if(f->d2_noisy_weights != NULL)
            copy_array(f->d2_noisy_weights,copy->d2_noisy_weights,in_out);// new
            if(f->d3_noisy_weights != NULL)
            copy_array(f->d3_noisy_weights,copy->d3_noisy_weights,in_out);// new
            if(f->d_noisy_biases != NULL)
            copy_array(f->d_noisy_biases,copy->d_noisy_biases,f->output);// new
            if(f->d1_noisy_biases != NULL)
            copy_array(f->d1_noisy_biases,copy->d1_noisy_biases,f->output);// new
            if(f->d2_noisy_biases != NULL)
            copy_array(f->d2_noisy_biases,copy->d2_noisy_biases,f->output);// new
            if(f->d3_noisy_biases != NULL)
            copy_array(f->d3_noisy_biases,copy->d3_noisy_biases,f->output);// new
        }// new
        if(f->d_biases != NULL)
        copy_array(f->d_biases,copy->d_biases,f->output);
        if(f->d1_biases != NULL)
        copy_array(f->d1_biases,copy->d1_biases,f->output);
        if(f->d2_biases != NULL)
        copy_array(f->d2_biases,copy->d2_biases,f->output);
        if(f->d3_biases != NULL)
        copy_array(f->d3_biases,copy->d3_biases,f->output);
    }
    if(exists_edge_popup_stuff_fcl(f) && exists_edge_popup_stuff_fcl(copy)){
        if(f->scores != NULL)
        copy_array(f->scores,copy->scores,in_out);
        if(f->d_scores != NULL)
        copy_array(f->d_scores,copy->d_scores,in_out);
        if(f->d1_scores != NULL)
        copy_array(f->d1_scores,copy->d1_scores,in_out);
        if(f->d2_scores != NULL)
        copy_array(f->d2_scores,copy->d2_scores,in_out);
        if(f->d3_scores != NULL)
        copy_array(f->d3_scores,copy->d3_scores,in_out);
        if(f->indices != NULL)
        copy_int_array(f->indices,copy->indices,in_out);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION && copy->normalization_flag == LAYER_NORMALIZATION){
        paste_bn(f->layer_norm,copy->layer_norm);
    }
    return;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_fcl_without_learning_parameters(fcl* f,fcl* copy){
    if(f == NULL || copy == NULL)
        return;
    if(f->output != copy->output)
        return;
    int in_out = f->output*f->input;
    if(in_out != copy->input*copy->output)
        return;
    copy->k_percentage = f->k_percentage;

    if(exists_d_params_fcl(f) && exists_d_params_fcl(copy)){
        copy_array(f->d_weights,copy->d_weights,in_out);
        if(is_noisy(f) && is_noisy(copy)){// new
            copy_array(f->d_noisy_weights,copy->d_noisy_weights,in_out);// new
            copy_array(f->d_noisy_biases,copy->d_noisy_biases,f->output);// new
        }// new
        copy_array(f->d_biases,copy->d_biases,f->output);
    }
    if(exists_edge_popup_stuff_fcl(f) && exists_edge_popup_stuff_fcl(copy)){
        copy_array(f->d_scores,copy->d_scores,in_out);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION && copy->normalization_flag == LAYER_NORMALIZATION){
        paste_bn_without_learning_parameters(f->layer_norm,copy->layer_norm);
    }
    return;
}


/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_w_fcl(fcl* f,fcl* copy){
    if(f == NULL)
        return;
    if(exists_params_fcl(f)){
        copy_array(f->weights,copy->weights,f->output*f->input);
        copy_array(f->biases,copy->biases,f->output);
        if(is_noisy(f) && is_noisy(copy)){// new
            copy_array(f->noisy_weights,copy->noisy_weights,f->output*f->input);// new
            copy_array(f->noisy_biases,copy->noisy_biases,f->output);// new
        }// new
    }
    if(exists_edge_popup_stuff_fcl(f)){
        copy_array(f->scores,copy->scores,f->input*f->output);
        copy_int_array(f->indices,copy->indices,f->input*f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        paste_w_bn(f->layer_norm,copy->layer_norm);
    return;
}

/* This function returns a fcl* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 *                @ float tau:= the tau param
 * */
void slow_paste_fcl(fcl* f,fcl* copy, float tau){
    if(f == NULL || copy == NULL)
        return;
    if(f->output != copy->output)
        return;
    int in_out = f->output*f->input;
    if(in_out != copy->input*copy->output)
        return;
    int i;
    for(i = 0; i < in_out; i++){
        if(i < f->output){
            if(exists_params_fcl(f) && exists_params_fcl(copy))
            copy->biases[i] = tau*f->biases[i] + (1-tau)*copy->biases[i];
            if(exists_d_params_fcl(f) && exists_d_params_fcl(copy)){
                if(f->d1_biases != NULL)
                copy->d1_biases[i] = tau*f->d1_biases[i] + (1-tau)*copy->d1_biases[i];
                if(f->d2_biases != NULL)
                copy->d2_biases[i] = tau*f->d2_biases[i] + (1-tau)*copy->d2_biases[i];
                if(f->d3_biases != NULL)
                copy->d3_biases[i] = tau*f->d3_biases[i] + (1-tau)*copy->d3_biases[i];
            }
        }
        if(exists_params_fcl(f) && exists_params_fcl(copy)){
            copy->weights[i] = tau*f->weights[i] + (1-tau)*copy->weights[i];
            if(is_noisy(f) && is_noisy(copy)){// new
                copy->noisy_weights[i] = tau*f->noisy_weights[i] + (1-tau)*copy->noisy_weights[i];// new
                if(i < f->output)
                copy->noisy_biases[i] = tau*f->noisy_biases[i] + (1-tau)*copy->noisy_biases[i];// new
            }// new
        }
        if(exists_d_params_fcl(f) && exists_d_params_fcl(copy)){
            if(f->d1_weights != NULL)
            copy->d1_weights[i] = tau*f->d1_weights[i] + (1-tau)*copy->d1_weights[i];
            if(f->d2_weights != NULL)
            copy->d2_weights[i] = tau*f->d2_weights[i] + (1-tau)*copy->d2_weights[i];
            if(f->d3_weights != NULL)
            copy->d3_weights[i] = tau*f->d3_weights[i] + (1-tau)*copy->d3_weights[i];
            if(is_noisy(f) && is_noisy(copy)){// new
                if(f->d1_noisy_weights != NULL)
                copy->d1_noisy_weights[i] = tau*f->d1_noisy_weights[i] + (1-tau)*copy->d1_noisy_weights[i];// new
                if(f->d2_noisy_weights != NULL)
                copy->d2_noisy_weights[i] = tau*f->d2_noisy_weights[i] + (1-tau)*copy->d2_noisy_weights[i];// new
                if(f->d3_noisy_weights != NULL)
                copy->d3_noisy_weights[i] = tau*f->d3_noisy_weights[i] + (1-tau)*copy->d3_noisy_weights[i];// new
                if(i < f->output){
                    if(f->d1_noisy_biases != NULL)
                    copy->d1_noisy_biases[i] = tau*f->d1_noisy_biases[i] + (1-tau)*copy->d1_noisy_biases[i];// new
                    if(f->d2_noisy_biases != NULL)
                    copy->d2_noisy_biases[i] = tau*f->d2_noisy_biases[i] + (1-tau)*copy->d2_noisy_biases[i];// new
                    if(f->d3_noisy_biases != NULL)
                    copy->d3_noisy_biases[i] = tau*f->d3_noisy_biases[i] + (1-tau)*copy->d3_noisy_biases[i];// new
                }
            }// new
        }
        if(exists_edge_popup_stuff_fcl(f) && exists_edge_popup_stuff_fcl(copy)){
            if(f->scores != NULL)
            copy->scores[i] = tau*f->scores[i] + (1-tau)*copy->scores[i];
            if(f->d1_scores != NULL)
            copy->d1_scores[i] = tau*f->d1_scores[i] + (1-tau)*copy->d1_scores[i];
            if(f->d2_scores != NULL)
            copy->d2_scores[i] = tau*f->d2_scores[i] + (1-tau)*copy->d2_scores[i];
            if(f->d3_scores != NULL)
            copy->d3_scores[i] = tau*f->d3_scores[i] + (1-tau)*copy->d3_scores[i];
        }
        
    }
    
    if(exists_edge_popup_stuff_fcl(copy) && exists_edge_popup_stuff_fcl(f)){
        for(i = 0; i < in_out; i++){
            if(copy->indices != NULL)
            copy->indices[i] = i;
        }
        if(copy->indices != NULL && copy->scores != NULL)
        sort(copy->scores,copy->indices,0,in_out-1);
        free(copy->active_output_neurons);
        copy->active_output_neurons = get_used_outputs(copy,NULL,FCLS,copy->output);
        
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION && copy->normalization_flag == LAYER_NORMALIZATION){
        slow_paste_bn(f->layer_norm,copy->layer_norm,tau);
    }
    return;
}

uint64_t count_weights_fcl(fcl* f){
    if(f == NULL)
        return 0;
    return (uint64_t)(f->input*f->output*f->k_percentage);
}
/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_params(fcl* f){
    if(f == NULL || !exists_params_fcl(f))
        return 0;
    uint64_t sum = 0;
    if(f->normalization_flag == LAYER_NORMALIZATION){
        sum += (uint64_t)f->layer_norm->vector_dim*2;
    }
    if(is_noisy(f)){// new
        sum+=(uint64_t)f->input*f->output;// new
        sum+=(uint64_t)f->output;// new
    }// new
    return (uint64_t)f->input*f->output+f->output+sum;
}


/* this function gives the number of float params for scores in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_scores_fcl(fcl* f){
    if(f == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return 0;
    return (uint64_t)f->input*f->output;
}


/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_weights(fcl* f){
    if(f == NULL || !exists_params_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return 0;
    uint64_t sum = 0;
    if(f->normalization_flag == LAYER_NORMALIZATION){
        sum += (uint64_t)f->layer_norm->vector_dim*2;
    }
    if(is_noisy(f)){// new
        sum+=(uint64_t)f->input*f->output;// new
    }// new
    return (uint64_t)f->input*f->output+sum;
}

/* this function pastes the weights and biases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params(fcl* f, float* vector){
    if(f == NULL || vector == NULL || !exists_params_fcl(f))
        return;
    
    int multiplier = 1;
    memcpy(f->weights,vector,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(f->noisy_weights,&vector[f->input*f->output],f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    memcpy(f->biases,&vector[multiplier*f->input*f->output],f->output*sizeof(float));// new
    if(is_noisy(f)){// new
        memcpy(f->noisy_biases,&vector[multiplier*f->input*f->output + f->output],f->output*sizeof(float));// new
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(f->layer_norm->gamma,&vector[multiplier*f->input*f->output+multiplier*f->output],f->layer_norm->vector_dim*sizeof(float));// new
        memcpy(f->layer_norm->beta,&vector[multiplier*f->input*f->output+multiplier*f->output + f->layer_norm->vector_dim],f->layer_norm->vector_dim*sizeof(float));// new
    }
}

/* this function pastes the scores stored in a vector inside a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_scores(fcl* f, float* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    memcpy(f->scores,vector,f->input*f->output*sizeof(float));
}

/* this function pastes the scores stored in a vector inside a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_indices2(fcl* f, int* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    memcpy(f->indices,vector,f->input*f->output*sizeof(int));
}

/* this function pastes the cl structure indices in a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ int* vector:= the vector where is copyed everything
 * */
void memcopy_indices_to_vector(fcl* f, int* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    memcpy(vector,f->indices,f->input*f->output*sizeof(int));    
    
}

/* this function pastes the scores stored in a vector inside a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void assign_vector_to_scores(fcl* f, float* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    f->scores=vector;
}

/* this function lets point to indices in a vector inside a fcl structure (only for edge popup)
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ int* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_indices(fcl* f, int* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    f->indices = vector;
}

/* this function pastes the the weights and biases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector(fcl* f, float* vector){
    if(f == NULL || !exists_params_fcl(f) || vector == NULL)
        return;
    int multiplier = 1;
    memcpy(vector,f->weights,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(&vector[f->input*f->output],f->noisy_weights,f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    memcpy(&vector[multiplier*f->input*f->output],f->biases,f->output*sizeof(float));// new
    if(is_noisy(f)){// new
        memcpy(&vector[multiplier*f->input*f->output + f->output],f->noisy_biases,f->output*sizeof(float));// new
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(&vector[multiplier*f->input*f->output+multiplier*f->output],f->layer_norm->gamma,f->layer_norm->vector_dim*sizeof(float));// new
        memcpy(&vector[multiplier*f->input*f->output+multiplier*f->output + f->layer_norm->vector_dim],f->layer_norm->beta,f->layer_norm->vector_dim*sizeof(float));// new
    }
}

/* this function pastes the scores from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_weights_to_vector(fcl* f, float* vector){
    if(f == NULL || !exists_params_fcl(f) || vector == NULL || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    int multiplier = 1;
    memcpy(vector,f->weights,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(&vector[f->input*f->output],f->noisy_weights,f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(&vector[multiplier*f->input*f->output],f->layer_norm->gamma,f->layer_norm->vector_dim*sizeof(float));// new
        memcpy(&vector[multiplier*f->input*f->output + f->layer_norm->vector_dim],f->layer_norm->beta,f->layer_norm->vector_dim*sizeof(float));// new
    }
}

/* this function pastes the the weights from vector to a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_weights(fcl* f, float* vector){
    if(f == NULL || !exists_params_fcl(f) || vector == NULL || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    int multiplier = 1;
    memcpy(f->weights,vector,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(f->noisy_weights,&vector[f->input*f->output],f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(f->layer_norm->gamma,&vector[multiplier*f->input*f->output],f->layer_norm->vector_dim*sizeof(float));// new
        memcpy(f->layer_norm->beta,&vector[multiplier*f->input*f->output + f->layer_norm->vector_dim],f->layer_norm->vector_dim*sizeof(float));// new
    }
}

/* this function pastes the scores from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_vector(fcl* f, float* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    memcpy(vector,f->scores,f->input*f->output*sizeof(float));
}

/* this function pastes the indices from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ int* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_indices(fcl* f, int* vector){
    if(f == NULL || vector == NULL || !exists_edge_popup_stuff_fcl(f) || f->feed_forward_flag == ONLY_DROPOUT)
        return;
    memcpy(vector,f->indices,f->input*f->output*sizeof(int));
}
/* this function pastes the dweights and dbiases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_derivative_params(fcl* f, float* vector){
    if(f == NULL || !exists_d_params_fcl(f) || vector == NULL)
        return;
    int multiplier = 1;
    memcpy(f->d_weights,vector,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(f->d_noisy_weights,&vector[f->input*f->output],f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    memcpy(f->d_biases,&vector[multiplier*f->input*f->output],f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(f->d_noisy_biases,&vector[multiplier*f->input*f->output + f->output],f->output*sizeof(float));// new
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(f->layer_norm->d_gamma,&vector[multiplier*f->input*f->output+multiplier*f->output],f->layer_norm->vector_dim*sizeof(float));
        memcpy(f->layer_norm->d_beta,&vector[multiplier*f->input*f->output+multiplier*f->output + f->layer_norm->vector_dim],f->layer_norm->vector_dim*sizeof(float));
    }
}


/* this function pastes the the dweights and dbiases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_derivative_params_to_vector(fcl* f, float* vector){
    if(f == NULL || !exists_d_params_fcl(f) || vector == NULL)
        return;
    int multiplier = 1;
    memcpy(vector,f->d_weights,f->input*f->output*sizeof(float));
    if(is_noisy(f)){// new
        memcpy(&vector[f->input*f->output],f->d_noisy_weights,f->input*f->output*sizeof(float));// new
        multiplier++;
    }// new
    memcpy(&vector[multiplier*f->input*f->output],f->d_biases,f->output*sizeof(float));// new
    if(is_noisy(f)){// new
        memcpy(&vector[multiplier*f->input*f->output+f->output],f->d_noisy_biases,f->output*sizeof(float));// new
    }// new
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(&vector[multiplier*f->input*f->output+multiplier*f->output],f->layer_norm->d_gamma,f->layer_norm->vector_dim*sizeof(float));// new
        memcpy(&vector[multiplier*f->input*f->output+multiplier*f->output + f->layer_norm->vector_dim],f->layer_norm->d_beta,f->layer_norm->vector_dim*sizeof(float));// new
    }
}

/* setting the biases to 0
 * Inpout:
 *             @ fcl* f:= the fully connected layer
 * */
void set_fully_connected_biases_to_zero(fcl* f){
    if(f == NULL || !exists_params_fcl(f))
        return;
    int i;
    for(i = 0; i < f->output; i++){
        f->biases[i] = 0;
    }
}

/* setting the unused weights to 0
 * Inpout:
 *             @ fcl* f:= the fully connected layer
 * */
void set_fully_connected_unused_weights_to_zero(fcl* f){
    if(f == NULL || f->indices == NULL || f->weights == NULL)
        return;
    int i;
    for(i = 0; i < f->output*f->input-f->output*f->input*f->k_percentage; i++){
        f->weights[f->indices[i]] = 0;
    }
}

/* this function sum up the scores in input1 and input2 in output
 * 
 * Input:
 * 
 * 
 *                 @ fcl* input1:= the first input fcl layer
 *                 @ fcl* input2:= the second input fcl layer
 *                 @ fcl* output:= the output fcl layer
 * */
void sum_score_fcl(fcl* input1, fcl* input2, fcl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    if(!exists_edge_popup_stuff_fcl(input1) || !exists_edge_popup_stuff_fcl(output) || !exists_edge_popup_stuff_fcl(input2))
        return;
    if(input1->input*input1->output != input2->input*input2->output || input1->input*input1->output != output->input*output->output)
        return;
    sum1D(input1->scores,input2->scores,output->scores,input1->input*input1->output);
}

/* this function stores in the output the best scores according to input1 and input2
 * 
 * Input:
 * 
 * 
 *                 @ fcl* input1:= the first input fcl layer
 *                 @ fcl* input2:= the second input fcl layer
 *                 @ fcl* output:= the output fcl layer
 * */
void compare_score_fcl(fcl* input1, fcl* input2, fcl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    if(!exists_edge_popup_stuff_fcl(input1) || !exists_edge_popup_stuff_fcl(output) || !exists_edge_popup_stuff_fcl(input2))
        return;
    int in_out = input1->input*input1->output;
    if(in_out != input2->input*input2->output || in_out != output->input*output->output)
        return;
    int i;
    for(i = 0; i < in_out; i++){
        if(input1->scores[i] > input2->scores[i] && bool_is_real(input1->scores[i]) && input1->scores[i] < MAXIMUM_SCORE)
            output->scores[i] = input1->scores[i];
        else if(bool_is_real(input2->scores[i]) && input2->scores[i] < MAXIMUM_SCORE)
            output->scores[i] = input2->scores[i];
    }
}

void compare_score_fcl_with_vector(fcl* input1, float* input2, fcl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    if(!exists_edge_popup_stuff_fcl(input1) || !exists_edge_popup_stuff_fcl(output))
        return;
    int in_out = input1->input*input1->output;
    if(in_out != in_out != output->input*output->output)
        return;
    int i;
    for(i = 0; i < in_out; i++){
        if(input1->scores[i] > input2[i] && bool_is_real(input1->scores[i]) && input1->scores[i] < MAXIMUM_SCORE)
            output->scores[i] = input1->scores[i];
        else if(bool_is_real(input2[i]) && input2[i] < MAXIMUM_SCORE)
            output->scores[i] = input2[i];
    }
}

/* this function divides the score with value
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fcl layer
 *                 @ float value:= the value that is gonna divide the scores
 * */
void dividing_score_fcl(fcl* f, float value){
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->scores[i]/=value;
    }
}

/* this function set the feed forward flag to only dropout checking the restriction needed
 * 
 * Input:
 * 
 * 
 *                 @ fcl* f:= the fully connected layer
 * 
 * */
void set_fcl_only_dropout(fcl* f){
    if(!f->dropout_flag){
        fprintf(stderr,"Error: if you use this layer only for dropout you should set dropout flag!\n");
        exit(1);
    }
    
    if(f->input!= f->output){
        fprintf(stderr,"Error: if you use only dropout then your input and output should match!\n");
        exit(1);
    }
    
    f->feed_forward_flag = ONLY_DROPOUT;
}


/* this function reset all the scores of the fcl layer to 0
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 * */
void reset_score_fcl(fcl* f){
    if(f == NULL || f->scores == NULL)
        return;
    if(f->feed_forward_flag == ONLY_DROPOUT)
        return;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->scores[i] = 0;
    }
    
}

/* thif function reinitialize the weights under the goodness function only if
 * they are among the f->input*f->output*percentage worst weights according to the scores
 * percentage and goodness should range in [0,1]
 * the re initialization uses the signed kaiming constant (the best one for edge popup according to the paper)
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 *                 @ float percentage:= the percentage of the worst weights
 *                 @ float goodness:= the goodness function
 * */
void reinitialize_weights_according_to_scores_fcl(fcl* f, float percentage, float goodness){
    if(f == NULL || !exists_edge_popup_stuff_fcl(f))
        return;
    int i;
    for(i = f->input*f->output-1;i > (int)(f->input*f->output*percentage); i--){
        if(f->scores[f->indices[i]] < goodness){
            f->weights[f->indices[i]] = signed_kaiming_constant(f->input);
            f->scores[f->indices[i]] = goodness;
            //if(is_noisy(f)){// new
            //    f->noisy_weights[f->indices[i]] = 0.5/(sqrtf(2*f->input));// new (not as the paper ,because different initialization for weights (signed kaiming constant)
            //}// new
        }
        else
            return;
    }
}

/* thif function reinitialize the weights under the goodness function only if
 * they are among the f->input*f->output*percentage worst weights according to the scores
 * percentage and goodness should range in [0,1]
 * the re initialization uses the signed kaiming constant (the best one for edge popup according to the paper)
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 *                 @ float percentage:= the percentage of the worst weights
 *                 @ float goodness:= the goodness function
 * */
void reinitialize_weights_according_to_scores_fcl_only_percentage(fcl* f, float percentage){
    if(f == NULL || !exists_edge_popup_stuff_fcl(f))
        return;
    if((int)(f->input*f->output*f->k_percentage) == 0)
        return;
    int val = 0;
    if((int)(f->input*f->output*f->k_percentage) == f->input*f->output)
        val = 1;
    int i;
    float score = f->scores[f->indices[(int)(f->input*f->output*f->k_percentage)-val]]-0.0000001;
    for(i = 0;i < (int)(f->input*f->output*percentage); i++){
        f->weights[f->indices[i]] = signed_kaiming_constant(f->input);
        f->scores[f->indices[i]] = score;
    }
}
/* thif function reinitialize the weights under the goodness function only if
 * they are among the f->input*f->output*percentage worst weights according to the scores
 * percentage and goodness should range in [0,1]
 * the re initialization uses the signed kaiming constant (the best one for edge popup according to the paper)
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 *                 @ float percentage:= the percentage of the worst weights
 *                 @ float goodness:= the goodness function
 * */
void reinitialize_weights_according_to_scores_and_inner_info_fcl(fcl* f){
    if(f == NULL || !exists_edge_popup_stuff_fcl(f))
        return;
    if((int)(f->input*f->output*f->k_percentage) == 0)
        return;
    float goodness = f->scores[f->indices[(int)(f->input*f->output*f->k_percentage)-1]];
    float percentage = f->k_percentage;
    reinitialize_weights_according_to_scores_fcl(f,percentage,goodness);
}

/* this function re initialize the weights and biases of the fully connected layers to get different values
 * 
 * Inputs:
 * 
 *             @ fcl* f:= the fully connected layers which bias and weights must be re initialized
 * */
void reinitialize_w_fcl(fcl* f){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->weights[i] = random_general_gaussian_xavier_init(f->input);
        if(is_noisy(f)){// new
            f->noisy_weights[i] = random_general_gaussian_xavier_init(f->input);// new
        }// new
    }
}

/* this function sets all the arrays needed for storing the partial derivatives and parameteres for sgd to 0
 * 
 * Inputs:
 * 
 * 
 *             @fcl* f:= the fully connected layer which arrays must be set to 0
 * */
fcl* reset_edge_popup_d_fcl(fcl* f){
    if (f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->d_scores[i] = 0;
        f->d1_scores[i] = 0;
        f->d2_scores[i] = 0;
        f->d3_scores[i] = 0;
    }
    return f;
}

/* this function set all the scores of the fcl layer to a low value (-99999)
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 * */
void set_low_score_fcl(fcl* f){
    if(f == NULL || f->scores == NULL)
        return;
    if(f->feed_forward_flag == ONLY_DROPOUT)
        return;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->scores[i] = -99999;
    }
    
}

/* this function returns an array that gives the used output*/
int* get_used_outputs(fcl* f, int* used_output, int flag, int output_size){
    int i,j;
    int* uo;
    if(used_output == NULL)
        uo= (int*)calloc(output_size,sizeof(int));
    else
        uo = used_output;
    
    for(i = 0; i < output_size; i++){
        uo[i] = 0;
    }
    
    for(i = f->input*f->output-f->input*f->output*f->k_percentage; i < f->input*f->output; i++){
        if(flag == CLS){
            int n_per_feature_map = f->output/output_size;
            for(j = 0; j < output_size; j++){
                if(((int)(f->indices[i]%f->output)< n_per_feature_map*(j+1) && (int)(f->indices[i]%f->output) >= n_per_feature_map*(j)))
                uo[j] = 1;
            }
        }
            
        else{
            uo[(int)((f->indices[i]/f->input))] = 1;
        }
    }
    
    return uo;  
    
}


void make_the_fcl_only_for_ff(fcl* f){
    if(f == NULL)
        return;
    // make it also for noisy layers
    free(f->d_weights);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->d3_weights);
    free(f->d_biases);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->d3_biases);
    free(f->temp);//output
    free(f->temp3);//output
    free(f->temp2);//input
    free(f->error2);//input
    f->d_weights = NULL;
    f->d1_weights = NULL;
    f->d2_weights = NULL;
    f->d3_weights = NULL;     
    f->d_biases = NULL;
    f->d1_biases = NULL;
    f->d2_biases = NULL;
    f->d3_biases = NULL;
    f->temp = NULL;
    f->temp2 = NULL;
    f->temp3 = NULL;
    f->error2 = NULL;
    f->training_mode = ONLY_FF;
    if(f->dropout_flag == DROPOUT){
        f->dropout_flag = DROPOUT_TEST;
        f->dropout_threshold = 1-f->dropout_threshold;
    }
}

void inference_fcl(fcl* f){
    if(f == NULL)
        return;
    if(f->dropout_flag == DROPOUT){
        f->dropout_flag = DROPOUT_TEST;
        f->dropout_threshold = 1-f->dropout_threshold;
    }
}

void eliminate_noisy_layers(fcl* f){
    if(is_noisy(f)){// new
        f->mode = STANDARD;// new
        free(f->temp_weights);// new
        free(f->noise);// new
        free(f->noisy_weights);// new
        free(f->d_noisy_weights);// new
        free(f->d1_noisy_weights);// new
        free(f->d2_noisy_weights);// new
        free(f->d3_noisy_weights);// new
        f->temp_weights = NULL;
        f->noise = NULL;
        f->noisy_weights = NULL;
        f->d_noisy_weights = NULL;
        f->d1_noisy_weights = NULL;
        f->d2_noisy_weights = NULL;
        f->d3_noisy_weights = NULL;
        free(f->temp_biases);// new
        free(f->noise_biases);// new
        free(f->noisy_biases);// new
        free(f->d_noisy_biases);// new
        free(f->d1_noisy_biases);// new
        free(f->d2_noisy_biases);// new
        free(f->d3_noisy_biases);// new
        f->temp_biases = NULL;
        f->noise_biases = NULL;
        f->noisy_biases = NULL;
        f->d_noisy_biases = NULL;
        f->d1_noisy_biases = NULL;
        f->d2_noisy_biases = NULL;
        f->d3_noisy_biases = NULL;
    }// new
}

void train_fcl(fcl* f){
    if(f == NULL)
        return;
    if(f->dropout_flag == DROPOUT_TEST){
        f->dropout_flag = DROPOUT;
        f->dropout_threshold = 1-f->dropout_threshold;
    }
}

void assign_noise_arrays(fcl* f, float** noise_biases, float** noise, int index){
    if(is_noisy(f)){
        float* noise_temp = f->noise;
        float* noise_biases_temp = f->noise_biases;
        f->noise = noise[index];
        f->noise_biases = noise_biases[index];
        noise[index] = noise_temp;
        noise_biases[index] = noise_biases_temp;
    }
    return;
}
