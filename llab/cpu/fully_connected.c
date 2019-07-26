/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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
    int i,j;
    for(j = 0; j < output_size; j++){
        for(i = 0; i < input_size; i++){
            output[j] += input[i]*weight[j*input_size+i];
        }
        output[j] += bias[j];
    }
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
void fully_connected_back_prop(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size){
    int i,j;
    for(j = 0; j < output_size; j++){
        for(i = 0; i < input_size; i++){
            weight_error[j*input_size+i] += output_error[j]*input[i];
            input_error[i] += output_error[j]*weight[j*input_size+i];
        }
        bias_error[j] += output_error[j];
    }
}



