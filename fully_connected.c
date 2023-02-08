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

/* This function computes the output of the current layer using the previous layer output
 * and the weights that connect the two layers and the biases of the output layer
 * 
 * Input:
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensions: input_size
 *         @ float* output:= a vector of outputs of the current layer, that must be filled
 *                           dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* bias:= a vector of bias of the current layer
 *                         dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output vector
 * */
void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size){
    if(bias != NULL){
        int i,j;
        for(j = 0; j < output_size; j++){
            for(i = 0; i < input_size; i++){
                output[j] += input[i]*weight[j*input_size+i];
            }
            output[j] += bias[j];
        }
    }
    else{
        int i,j;
        for(j = 0; j < output_size; j++){
            for(i = 0; i < input_size; i++){
                output[j] += input[i]*weight[j*input_size+i];
            }
        }
    }
}

/* This function computes the output of the current layer using the previous layer output
 * and the weights that connect the two layers and the biases of the output layer
 * 
 * Input:
 *            @ float* noise:= where the noise will be generated and stored
 *            @ float* new_weights:= where the new weights that will be multiplied by the input will be stored new_weights = noisy_weight*random_normal+weights
 *            @ float* noisy_weights:= the weights for the noise
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensions: input_size
 *         @ float* output:= a vector of outputs of the current layer, that must be filled
 *                           dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* bias:= a vector of bias of the current layer
 *                         dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output vector
 * */
void noisy_fully_connected_feed_forward(float* noise_biases, float* new_biases,float* noisy_biases, float* noise, float* new_weights,float* noisy_weights, float* input, float* output, float* weight,float* bias, int input_size, int output_size){
    int size = output_size*input_size;
    int i,j;
    for(i = 0; i < size; i++){
        new_weights[i] = noisy_weights[i]*noise[i]+weight[i];
    }
    
    
    if(bias != NULL){
        for(j = 0; j < output_size; j++){
			new_biases[j] = noisy_biases[j]*noise_biases[j]+bias[j];
            for(i = 0; i < input_size; i++){
                output[j] += input[i]*new_weights[j*input_size+i];
            }
            output[j] += new_biases[j];
        }
    }
    else{
        for(j = 0; j < output_size; j++){
            for(i = 0; i < input_size; i++){
                output[j] += input[i]*new_weights[j*input_size+i];
            }
        }
    }
}

/* This function computes the output of the current layer using the previous layer output
 * and the weights that connect the two layers and the biases of the output layer, is used by the edge popup feed forward
 * 
 * Input:
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensions: input_size
 *         @ float* output:= a vector of outputs of the current layer, that must be filled
 *                           dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* bias:= a vector of bias of the current layer
 *                         dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output vector
 *         @ int* indices:= the indices of the weigths sorted by the scores of the weights
 *         @ int last_n:= the last_n indices are the indices of the best last_n weights according to their scores
 * */
void fully_connected_feed_forward_edge_popup(float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, int last_n){
    int i,j;
    
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        output[(int)(indices[j]/input_size)] += input[(indices[j]%input_size)]*weight[indices[j]];
    }
    /*
    if(bias != NULL){
        for(j = 0; j < output_size; j++){
            output[j] += bias[j];
        }
    }
    * */
}
/* This function computes the output of the current layer using the previous layer output
 * and the weights that connect the two layers and the biases of the output layer, is used by the edge popup feed forward
 * 
 * Input:
 *         @ float* noise:= where the noise will be generated and stored
 *            @ float* new_weights:= where the new weights that will be multiplied by the input will be stored new_weights = noisy_weight*random_normal+weights
 *            @ float* noisy_weights:= the weights for the noise
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensions: input_size
 *         @ float* output:= a vector of outputs of the current layer, that must be filled
 *                           dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* bias:= a vector of bias of the current layer
 *                         dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output vector
 *         @ int* indices:= the indices of the weigths sorted by the scores of the weights
 *         @ int last_n:= the last_n indices are the indices of the best last_n weights according to their scores
 * */
void noisy_fully_connected_feed_forward_edge_popup(float* noise, float* new_weights,float* noisy_weights, float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, int last_n){
    int i,j;
    
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        new_weights[indices[j]] = noisy_weights[indices[j]]*noise[indices[j]]+weight[indices[j]];
        output[(int)(indices[j]/input_size)] += input[(indices[j]%input_size)]*new_weights[indices[j]];
    }
    /*
    if(bias != NULL){
        for(j = 0; j < output_size; j++){
            output[j] += bias[j];
        }
    }
    * */
}

/* This function computes the error of the previous layer and the error of the weights and biases
 * using the current output layer error and the weights that connect the two layers
 * 
 * Input:
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensione: input_size
 *         @ float* output_error:= a vector of the errors of the current layer
 *                                 dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* input_error:= a vector of error of the previous layer that must be filled
 *                                dimensions: input_size
 *         @ float* weight_error:= a vector of error of the of the weights of the two layers that must be filled
 *                                 dimensions: output_size*input_size
 *         @ float* bias_error:= a vector of error of the of the biases of the current layer that must be filled
 *                               dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output_error vector
 * */
void fully_connected_back_prop(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,int training_flag){
    if(bias_error != NULL){
        int i,j;
        for(j = 0; j < output_size; j++){
            for(i = 0; i < input_size; i++){
                weight_error[j*input_size+i] += output_error[j]*input[i];
                input_error[i] += output_error[j]*weight[j*input_size+i];
            }
            bias_error[j] += output_error[j];
        }
    }
    else{
        if(training_flag == FREEZE_TRAINING){
            int i,j;
            for(j = 0; j < output_size; j++){
                for(i = 0; i < input_size; i++){
                    input_error[i] += output_error[j]*weight[j*input_size+i];
                }
            }
        }
        else{
            int i,j;
            for(j = 0; j < output_size; j++){
                for(i = 0; i < input_size; i++){
                    weight_error[j*input_size+i] += output_error[j]*input[i];
                    input_error[i] += output_error[j]*weight[j*input_size+i];
                }
            }
        }
    }
}

/* This function computes the error of the previous layer and the error of the weights and biases
 * using the current output layer error and the weights that connect the two layers
 * 
 * Input:
 *            @ float* noise:= where the noise will be generated and stored
 *            @ float* new_weights:= where the new weights that will be multiplied by the input will be stored new_weights = noisy_weight*random_normal+weights
 *            @ float* noisy_weights:= the weights for the noise
 *         @ float* noisy_weights_error:= where the partial derivatives for noisy weights are stored
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensione: input_size
 *         @ float* output_error:= a vector of the errors of the current layer
 *                                 dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* input_error:= a vector of error of the previous layer that must be filled
 *                                dimensions: input_size
 *         @ float* weight_error:= a vector of error of the of the weights of the two layers that must be filled
 *                                 dimensions: output_size*input_size
 *         @ float* bias_error:= a vector of error of the of the biases of the current layer that must be filled
 *                               dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output_error vector
 * */
void noisy_fully_connected_back_prop(float* noise_biases, float* new_biases,float* noisy_biases,float* noisy_biases_error, float* noise, float* new_weights,float* noisy_weights, float* noisy_weights_error, float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,int training_flag){
    if(bias_error != NULL){
        int i,j;
        for(j = 0; j < output_size; j++){
            for(i = 0; i < input_size; i++){
                weight_error[j*input_size+i] += output_error[j]*input[i];
                input_error[i] += output_error[j]*new_weights[j*input_size+i];
                noisy_weights_error[j*input_size+i] += weight_error[j*input_size+i]*noise[j*input_size+i];
                
            }
            bias_error[j] += output_error[j];
            noisy_biases_error[j]+=output_error[j]*noise_biases[j];
        }
    }
    else{
        if(training_flag == FREEZE_TRAINING){
            int i,j;
            for(j = 0; j < output_size; j++){
                for(i = 0; i < input_size; i++){
                    input_error[i] += output_error[j]*weight[j*input_size+i];
                }
            }
        }
        else{
            int i,j;
            for(j = 0; j < output_size; j++){
                for(i = 0; i < input_size; i++){
                    weight_error[j*input_size+i] += output_error[j]*input[i];
                    input_error[i] += output_error[j]*weight[j*input_size+i];
                    noisy_weights_error[j*input_size+i] += weight_error[j*input_size+i]*noise[j*input_size+i];
                }
            }
        }
    }
}

/* This function computes the error of the previous layer and the error of the weights and biases
 * using the current output layer error and the weights that connect the two layers, gradient descent = edge popup
 * 
 * Input:
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensione: input_size
 *         @ float* output_error:= a vector of the errors of the current layer
 *                                 dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* input_error:= a vector of error of the previous layer that must be filled
 *                                dimensions: input_size
 *         @ float* weight_error:= a vector of error of the of the weights of the two layers that must be filled
 *                                 dimensions: output_size*input_size
 *         @ float* bias_error:= a vector of error of the of the biases of the current layer that must be filled
 *                               dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output_error vector
 *            @ float* score_errors:= the errors of the score
 *            @ int* indices:= the indices of the weights sorted by the scores
 *            @ int last_n:= the last n best weights
 * */
 
void fully_connected_back_prop_edge_popup(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n){
    int i,j;
    for(j = 0; j < output_size; j++){
        for(i = 0; i < input_size; i++){
            score_error[j*input_size+i] += output_error[j]*input[i]*weight[j*input_size+i];
        }
    }

    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        input_error[(indices[j]%input_size)] += output_error[(int)(indices[j]/input_size)]*weight[(indices[j])];
    }
}

/* This function computes the error of the previous layer and the error of the weights and biases
 * using the current output layer error and the weights that connect the two layers, gradient descent = edge popup
 * 
 * Input:
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensione: input_size
 *         @ float* output_error:= a vector of the errors of the current layer
 *                                 dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* input_error:= a vector of error of the previous layer that must be filled
 *                                dimensions: input_size
 *         @ float* weight_error:= a vector of error of the of the weights of the two layers that must be filled
 *                                 dimensions: output_size*input_size
 *         @ float* bias_error:= a vector of error of the of the biases of the current layer that must be filled
 *                               dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output_error vector
 *            @ float* score_errors:= the errors of the score
 *            @ int* indices:= the indices of the weights sorted by the scores
 *            @ int last_n:= the last n best weights
 * */
void fully_connected_back_prop_edge_popup_ff_gd_bp(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n){
    int i,j;
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        weight_error[(indices[j])]+=output_error[((int)(indices[j]/input_size))]*input[(indices[j]%input_size)];
    }
    /*
    if(bias_error != NULL){
        for(j = 0; j < output_size; j++){
            bias_error[j]+=output_error[j];
        }
    }
    */
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        input_error[(indices[j]%input_size)] += output_error[((int)(indices[j]/input_size))]*weight[indices[j]];
    }
}
/* This function computes the error of the previous layer and the error of the weights and biases
 * using the current output layer error and the weights that connect the two layers, gradient descent = edge popup
 * 
 * Input:
 *            @ float* noise:= where the noise will be generated and stored
 *            @ float* new_weights:= where the new weights that will be multiplied by the input will be stored new_weights = noisy_weight*random_normal+weights
 *            @ float* noisy_weights:= the weights for the noise
 *         @ float* noisy_weights_error:= where the partial derivatives for noisy weights are stored
 *         @ float* input:= a vector of inputs of the previous layer
 *                          dimensione: input_size
 *         @ float* output_error:= a vector of the errors of the current layer
 *                                 dimensions: output_size
 *         @ float* weight:= a vector of weight which connects the current layer with the prvious one
 *                           dimensions: output_size*input_size
 *         @ float* input_error:= a vector of error of the previous layer that must be filled
 *                                dimensions: input_size
 *         @ float* weight_error:= a vector of error of the of the weights of the two layers that must be filled
 *                                 dimensions: output_size*input_size
 *         @ float* bias_error:= a vector of error of the of the biases of the current layer that must be filled
 *                               dimensions: output_size
 *         @ int input_size:= the size of the float* input vector
 *         @ int output_size:= the size of the float* output_error vector
 *            @ float* score_errors:= the errors of the score
 *            @ int* indices:= the indices of the weights sorted by the scores
 *            @ int last_n:= the last n best weights
 * */
void noisy_fully_connected_back_prop_edge_popup_ff_gd_bp(float* noise, float* new_weights,float* noisy_weights, float* noisy_weights_error,float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n){
    int i,j;
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        weight_error[(indices[j])]+=output_error[((int)(indices[j]/input_size))]*input[(indices[j]%input_size)];
        noisy_weights_error[(indices[j])]+=weight_error[(indices[j])]*noise[(indices[j])];
    }
    /*
    if(bias_error != NULL){
        for(j = 0; j < output_size; j++){
            bias_error[j]+=output_error[j];
        }
    }
    */
    for(j = output_size*input_size-last_n; j < output_size*input_size; j++){
        input_error[(indices[j]%input_size)] += output_error[((int)(indices[j]/input_size))]*new_weights[indices[j]];
    }
}




