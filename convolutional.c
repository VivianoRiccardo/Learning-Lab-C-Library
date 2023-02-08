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

/* This function computes the feed forwad of a feature map using the previous convolutional layer
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output:= the current feature map computed using the input, kernel and bias
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1, int stride2, int padding){
    int oi,oj,i,j,c;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(c = 0; c < channels; c++){
                for(i = 0; i < kernel_i; i++){
                    for(j = 0; j < kernel_j; j++){
                        output[oi*output_j+oj] += kernel[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
                    }
                }
            }
            output[oi*output_j+oj] += bias;    
        }
    }
}

/* This function computes the feed forwad of a feature map using the convolution through a subset of all the weights (edge-popup algorithm)
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float** kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float* bias:= the total biases of the output, dimension: n_kernel
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output:= the final total output
 *                               dimensions: n_kernel*((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 *             @ int* indices:= the array indices of the weights sorted by the score
 *             @ int n_kernel:= the number of kernels
 *             @ int last_n:= the last n best indices
 * */
void convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n){
    
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(s = n_kernels-last_n; s < n_kernels; s++){
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            output[indices[s]*output_i*output_j + oi*output_j+oj] += kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
                        }
                    }
                }
                output[indices[s]*output_i*output_j + oi*output_j+oj] += bias[indices[s]];    
            }
        }
    }
    */
    
    
    int flagg = 0;
    int flagg2 = 0;
    
    int oi,oj,i,j,c,s,z;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    int lv_minus1 = n_kernels*channels*kernel_i*kernel_j-last_n;
    int lv0 = n_kernels*channels*kernel_i*kernel_j;
    int lv1 = channels*kernel_i*kernel_j;
    int lv2 = kernel_i*kernel_j;
    int ou_lv1 = output_i*output_j;
    int in_lv1 = input_i*input_j;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(s = lv_minus1; s < lv0; s++){
                
                int kernel_temp = ((int)(indices[s]/(lv1)));
                int rest_temp = ((indices[s]%(lv1)));
                int c_temp = (int)(rest_temp/lv2);
                int i_temp = (int)((rest_temp%lv2)/kernel_j);
                int j_temp = ((rest_temp%lv2)%kernel_j);
                //if(input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j] != 0)
                //printf("one: %f, two: %f, three: %f\n",kernel[(int)(indices[s]/(channels*kernel_i*kernel_j))][((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*kernel_i*kernel_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*kernel_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)],input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j],kernel[(int)(indices[s]/(channels*kernel_i*kernel_j))][((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*kernel_i*kernel_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*kernel_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)]*input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j]);
                output[kernel_temp*ou_lv1+oi*output_j+oj] += kernel[kernel_temp][rest_temp]*input[c_temp*input_i*input_j + i_temp*input_j + j_temp+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
               
            }
        }
    }
    
    /* mmh pretty useless the biases right?
     * they are not because you can fine tune the output based on the bias
     * however, the training of the edge popup is done through the weights
     * and not biases. The biases are not trained, but most initlization
     * of biases is with 0s. so is pretty useless this computation.
     * If you want to have the biases too, de comment this part and also look at
     * the backpropagation part below
     * 
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(i = 0; i < n_kernels; i++){
                output[i*output_i*output_j+oi*output_j+oj]+=bias[i]; 
            }
        }
    }
    */
}

/* This function computes the errors for convolution using the backpropagation
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding){
    int oi,oj,i,j,c;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    // don't avoid biases computation
    if(bias_error != NULL){
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            kernel_error[c*kernel_i*kernel_j + i*kernel_j + j] += output_error[oi*output_j+oj]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
                            input_error[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j] += kernel[c*kernel_i*kernel_j + i*kernel_j + j]*output_error[oi*output_j+oj];
                        }
                    }
                }
                (*bias_error) += output_error[oi*output_j+oj];
            }
        }
    }
    // avoid biases computation
    else{
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            kernel_error[c*kernel_i*kernel_j + i*kernel_j + j] += output_error[oi*output_j+oj]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
                            input_error[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j] += kernel[c*kernel_i*kernel_j + i*kernel_j + j]*output_error[oi*output_j+oj];
                        }
                    }
                }
            }
        }
    }
}


/* This function computes the errors of a convolution using the backpropagation for a subset of weights (edge popup)
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 *             @ float* score_error:= the error that must be computed
 * */
void convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error){
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*int oi,oj,i,j,c;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(c = 0; c < channels; c++){
                for(i = 0; i < kernel_i; i++){
                    for(j = 0; j < kernel_j; j++){
                        (*score_error) += output_error[oi*output_j+oj]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j]*kernel[c*kernel_i*kernel_j + i*kernel_j + j];
                    }
                }
            }
        }
    }
    */
    //printf("conv_back_propo edge popup\n");
    int oi,oj,i,j,c;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(c = 0; c < channels; c++){
                for(i = 0; i < kernel_i; i++){
                    for(j = 0; j < kernel_j; j++){
                        score_error[c*kernel_i*kernel_j + i*kernel_j + j] += output_error[oi*output_j+oj]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j]*kernel[c*kernel_i*kernel_j + i*kernel_j + j];
                        

                    }
                }
            }
        }
    }

}

/* This function should be used when the network uses the edge popup for the feed forward, and the gradient descent for the back propagation.
 * Basically you should train a network with edge popup, finding the optimal subnetwork, then you can train it more adjusting the weights and biases
 * In this computation the biases are considered, so you need to de comment the part above
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float* bias:= the total biases of the output
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= total output error
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float** kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: n_kernelsXchannels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: n_kernel
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 *             @ float* score_error:= the error that must be computed
 *             @ int* indices:= the indices of the scores: dimensions: n_kernels*n_channels*kernel_rows*kernel_cols
 *             @ int last_n:= the last_n used weights
 * */
void convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output_error, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error){
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(s = n_kernels-last_n; s < n_kernels; s++){
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            kernel_error[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j] += output_error[indices[s]*output_i*output_j + oi*output_j+oj]*input[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j];
                        }
                    }
                }
                bias_error[indices[s]] += output_error[indices[s]*output_i*output_j + oi*output_j+oj];    
            }
        }
    }
    */
    int oi,oj,i,j,c,s,z;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                kernel_error[(int)(indices[s]/(channels*kernel_i*kernel_j))][((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*kernel_i*kernel_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*kernel_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)] += output_error[((int)(indices[s]/(channels*kernel_i*kernel_j)))*output_i*output_j+oi*output_j+oj]*input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j];            
            }
        }
    }
    if(bias_error != NULL){
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(i = 0; i < n_kernels; i++){
                    bias_error[i]+=output_error[i*output_i*output_j+oi*output_j+oj]; 
                }
            }
        }
    }
    
    
}

/* This function computes the errors using the backpropagation of the edge popup. It considers the error only for the inputs
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: n_kernels*((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 *             @ float* score_error:= the error that must be computed
 * */
void convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n){
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    for(s = n_kernels-last_n; s < n_kernels; s++){
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            input_error[c*input_i*input_j + i*input_j + j+(oj-padding)*stride2+(oi-padding)*stride1*input_j] += output_error[indices[s]*output_i*output_j + oi*output_j+oj]*kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j];
                        }
                    }
                }
            }
        }
    }
    */
    int oi,oj,i,j,c,s,z;
    int output_i = (input_i-kernel_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride2 + 1 + 2*padding;
    int lv1 = channels*kernel_i*kernel_j;
    int lv2 = kernel_i*kernel_j;
    int ou_lv1 = output_i*output_j;
    int in_lv1 = input_i*input_j;
    int s1 = n_kernels*channels*kernel_i*kernel_j-last_n;
    int s2 = n_kernels*channels*kernel_i*kernel_j;
    for(oi = padding; oi < output_i-padding; oi++){
        for(oj = padding; oj < output_j-padding; oj++){
            for(s = s1; s < s2; s++){
                int kernel_temp = ((int)(indices[s]/(lv1)));
                int rest_temp = ((indices[s]%(lv1)));
                int c_temp = (int)(rest_temp/lv2);
                int i_temp = (int)((rest_temp%lv2)/kernel_j);
                int j_temp = ((rest_temp%lv2)%kernel_j);
                //if(input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j] != 0)
                //printf("one: %f, two: %f, three: %f\n",kernel[(int)(indices[s]/(channels*kernel_i*kernel_j))][((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*kernel_i*kernel_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*kernel_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)],input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j],kernel[(int)(indices[s]/(channels*kernel_i*kernel_j))][((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*kernel_i*kernel_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*kernel_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)]*input[((int)((indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j)))*input_i*input_j + ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j))*input_j + (((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j)+(oj-padding)*stride2+(oi-padding)*stride1*input_j]);
                input_error[c_temp*input_i*input_j + i_temp*input_j + j_temp+(oj-padding)*stride2+(oi-padding)*stride1*input_j]+=output_error[kernel_temp*ou_lv1+oi*output_j+oj]*kernel[kernel_temp][rest_temp];
                
            }
        }
    }
    
}
/* This function apply the 2D max-pooling to a covolutional layer
 * 
 * Input:
 *             @ float* input:= a feature map to which the pooling is applied
 *                              dimensions: input_i*input_j
 *             @ float* output:= the output computed after applying the pooling to the input
 *                               dimensions: ((input_i-sub_pool_i)/stride + 1 + 2*padding)*((input_j-sub_pool_j)/stride + 1 + 2*padding)
 *             @ int input_i:= the rows of the feature map input
 *             @ int input_j:= the number of columns of the feature map output
 *             @ int sub_pool_i:= the number of rows used for each pooling iteration
 *             @ int sub_pool_j:= the number of columns used for each pooling iteration
 *             @ int stride1:= the stride used to pool for the rows
 *             @ int stride2:= the stride used to pool for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void max_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding){
    int i,j,k1,k2;
    int output_i = (input_i-sub_pool_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-sub_pool_j)/stride2 + 1 + 2*padding;
    float max = -999999;
    
    for(i = 0; i < output_i - 2*padding; i++){
        for(j = 0; j < output_j - 2*padding; j++){
            for(k1 = 0; k1 < sub_pool_i; k1++){
                for(k2 = 0; k2 < sub_pool_j; k2++){                   
                    if(input[input_j*(i*stride1+k1) + j*stride2 + k2] > max)
                        max = input[input_j*(i*stride1+k1) + j*stride2 + k2];
                }
            }
            output[(padding+i)*output_j+padding+j] = max;
            max = -999999;        
        }
    }
    
     
}

/* This function computed the error for a max-pool layer
 * 
 * Input:
 *             @ float* input:= a feature map to which the pooling is applied
 *                              dimensions: input_i*input_j
 *             @ float* output_error:= the output_error used to compute the input error
 *                               dimensions: ((input_i-sub_pool_i)/stride + 1 + 2*padding)*((input_j-sub_pool_j)/stride + 1 + 2*padding)
 *             @ int input_i:= the rows of the feature map input
 *             @ int input_j:= the number of columns of the feature map output
 *             @ int sub_pool_i:= the number of rows used for each pooling iteration
 *             @ int sub_pool_j:= the number of columns used for each pooling iteration
 *             @ int stride1:= the stride used to pool for the rows
 *             @ int stride2:= the stride used to pool for the cols
 *             @ int padding:= the optional padding added to the output
 *             @ float input_error := the error computed using the output_error
 *                                    dimensions: input_i*input_j
 * */
void max_pooling_back_prop(float* input, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding, float* input_error){
    int i,j,k1,k2, index1 = 0, index2 = 0;
    int output_i = (input_i-sub_pool_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-sub_pool_j)/stride2 + 1 + 2*padding;
    float max = -999999;
    
    for(i = 0; i < output_i - 2*padding; i++){
        for(j = 0; j < output_j - 2*padding; j++){
            for(k1 = 0; k1 < sub_pool_i; k1++){
                for(k2 = 0; k2 < sub_pool_j; k2++){
                    
                    if(input[input_j*(i*stride1+k1) + j*stride2 + k2] > max){
                        max = input[input_j*(i*stride1+k1) + j*stride2 + k2];
                        index1 = k1;
                        index2 = k2;
                        
                    }
                
                }
            }
            
            for(k1 = 0; k1 < sub_pool_i; k1++){
                for(k2 = 0; k2 < sub_pool_j; k2++){
                    
                    if(index1 == k1 && index2 == k2)
                        input_error[input_j*(i*stride1+k1) + j*stride2 + k2] = output_error[(padding+i)*output_j+padding+j];
                    
                    else
                        input_error[input_j*(i*stride1+k1) + j*stride2 + k2] = 0;
                        
                
                }
            }
            max = -9999999;        
        }
    } 
}

/* This function apply the 2D avarage-pooling to a covolutional layer
 * 
 * Input:
 *             @ float* input:= a feature map to which the pooling is applied
 *                              dimensions: input_i*input_j
 *             @ float* output:= the output computed after applying the pooling to the input
 *                               dimensions: ((input_i-sub_pool_i)/stride + 1 + 2*padding)*((input_j-sub_pool_j)/stride + 1 + 2*padding)
 *             @ int input_i:= the rows of the feature map input
 *             @ int input_j:= the number of columns of the feature map output
 *             @ int sub_pool_i:= the number of rows used for each pooling iteration
 *             @ int sub_pool_j:= the number of columns used for each pooling iteration
 *             @ int stride1:= the stride used to pool for the rows
 *             @ int stride2:= the stride used to pool for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void average_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding){
    int i,j,k1,k2;
    int output_i = (input_i-sub_pool_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-sub_pool_j)/stride2 + 1 + 2*padding;
    float sum = 0;
    
    for(i = 0; i < output_i - 2*padding; i++){
        for(j = 0; j < output_j - 2*padding; j++){
            for(k1 = 0; k1 < sub_pool_i; k1++){
                for(k2 = 0; k2 < sub_pool_j; k2++){
                    sum += input[input_j*(i*stride1+k1) + j*stride2 + k2];
                }
            }
            output[(padding+i)*output_j+padding+j] = sum/(sub_pool_i*sub_pool_j);
            sum = 0;        
        }
    }
}

/* This function computed the error for an avarage-pool layer
 * 
 * Input:
 *            @ float input_error := the error computed using the output_error
 *                                    dimensions: input_i*input_j
 *             @ float* output_error:= the output_error used to compute the input error
 *                               dimensions: ((input_i-sub_pool_i)/stride + 1 + 2*padding)*((input_j-sub_pool_j)/stride + 1 + 2*padding)
 *             @ int input_i:= the rows of the feature map input
 *             @ int input_j:= the number of columns of the feature map output
 *             @ int sub_pool_i:= the number of rows used for each pooling iteration
 *             @ int sub_pool_j:= the number of columns used for each pooling iteration
 *             @ int stride1:= the stride used to pool for the rows
 *             @ int stride2:= the stride used to pool for the cols
 *             @ int padding:= the optional padding added to the output
 *             
 * */
void average_pooling_back_prop(float* input_error, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding){
    int i,j,k1,k2;
    int output_i = (input_i-sub_pool_i)/stride1 + 1 + 2*padding;
    int output_j = (input_j-sub_pool_j)/stride2 + 1 + 2*padding;
    
    for(i = 0; i < output_i - 2*padding; i++){
        for(j = 0; j < output_j - 2*padding; j++){
            for(k1 = 0; k1 < sub_pool_i; k1++){
                for(k2 = 0; k2 < sub_pool_j; k2++){
                    input_error[input_j*(i*stride1+k1) + j*stride2 + k2] = ((float)1/(sub_pool_i*sub_pool_j))*output_error[(padding+i)*output_j+padding+j];
                }
            }
        }
    }
}


/* This function computes the feed forwad of a feature map using the previous teansposed convolutional layer
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output:= the current feature map computed using the input, kernel and bias
 *                               dimensions: ((input_i-1)*stride+kernel_i-2*padding)*((input_j-1)*stride+kernel_j-2*padding)
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1,int stride2, int padding){
    int oi,oj,i,j,c;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            output[i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[c*kernel_i*kernel_j + i*kernel_j + j];
                        }
                    }
                }    
            }
        }
        for(oi = 0; oi < output_i*output_j; oi++){
            output[oi]+=bias;
        }
    }
    else if(padding){
        float* temp = (float*)calloc(output_i*output_j,sizeof(float));
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            temp[i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[c*kernel_i*kernel_j + i*kernel_j + j];
                        }
                    }
                }    
            }
        }
        
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                output[(oi-padding)*(output_j-2*padding)+oj-padding] = temp[oi*output_j+oj]+bias;
            }
        }
        free(temp);
    }
}


/* This function computes the feed forwad of a feature map using the previous teansposed convolutional layer
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output:= the current feature map computed using the input, kernel and bias
 *                               dimensions: ((input_i-1)*stride+kernel_i-2*padding)*((input_j-1)*stride+kernel_j-2*padding)
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1,int stride2, int padding, int* indices, int n_kernels, int last_n){
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                output[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j];
                            }
                        }
                    }    
                }
            }
        }
            for(oi = 0; oi < output_i*output_j; oi++){
                output[indices[s]*output_i*output_j + oi]+=bias[indices[s]];
            }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                temp[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j];
                            }
                        }
                    }    
                }
            }
        }
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    output[indices[s]*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding] = temp[indices[s]*output_i*output_j + oi*output_j+oj]+bias[indices[s]];
                }
            }
        }
        free(temp);
    }
    * */
    
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    output[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[k][c*kernel_i*kernel_j + i*kernel_j + j];
                }
            }
        }
        for(i = 0; i < n_kernels; i++){
            for(oi = 0; oi < output_i*output_j; oi++){
                output[i*output_i*output_j + oi]+=bias[i];
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    temp[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]+=input[c*input_i*input_j + oi*input_j+oj]*kernel[k][c*kernel_i*kernel_j + i*kernel_j + j];       
                }    
            }
        }

        for(i = 0; i < n_kernels; i++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    output[i*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding] = temp[i*output_i*output_j + oi*output_j+oj];
                }
            }
        }
        /*
        for(i = 0; i < n_kernels; i++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    output[i*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding] += bias[i];
                }
            }
        }*/
        free(temp);
    }
    
    
}

/* This function computes the errors using the backpropagation for transposed convolution
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding){
    int oi,oj,i,j,c;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            input_error[c*input_i*input_j + oi*input_j+oj]+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*output_error[i*output_j + j+oj*stride2+oi*stride1*output_j];
                            kernel_error[c*kernel_i*kernel_j + i*kernel_j + j]+=input[c*input_i*input_j + oi*input_j+oj]*output_error[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }
        if(bias_error != NULL){
            for(oi = 0; oi < output_i*output_j; oi++){
                (*bias_error)+=output_error[oi];
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(output_i*output_j,sizeof(float));
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                temp[oi*output_j+oj] = output_error[(oi-padding)*(output_j-2*padding)+oj-padding];
            }
        }
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            input_error[c*input_i*input_j + oi*input_j+oj]+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*temp[i*output_j + j+oj*stride2+oi*stride1*output_j];
                            kernel_error[c*kernel_i*kernel_j + i*kernel_j + j]+=input[c*input_i*input_j + oi*input_j+oj]*temp[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }
        if(bias_error != NULL){
            for(oi = 0; oi < output_i*output_j; oi++){
                (*bias_error)+=temp[oi];
            }
        }
        
        
        free(temp);
    }
}


/* This function computes the errors using the backpropagation for transposed convolution
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error){
    // this commented part is in the case we are considering the scores not as high as the number of kernels*kernl_rows*kernel_cols*channels
    // but as the number of kernel only. This means that indices dimension is n_kernels
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            (*score_error)+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + oi*input_j+oj]*output_error[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(output_i*output_j,sizeof(float));
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                temp[oi*output_j+oj] = output_error[(oi-padding)*(output_j-2*padding)+oj-padding];
            }
        }
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            (*score_error)+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + oi*input_j+oj]*temp[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }
        
        free(temp);
    }
    * */
    
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            score_error[c*kernel_i*kernel_j + i*kernel_j + j]+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + oi*input_j+oj]*output_error[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(output_i*output_j,sizeof(float));
        for(oi = padding; oi < output_i-padding; oi++){
            for(oj = padding; oj < output_j-padding; oj++){
                temp[oi*output_j+oj] = output_error[(oi-padding)*(output_j-2*padding)+oj-padding];
            }
        }
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(c = 0; c < channels; c++){
                    for(i = 0; i < kernel_i; i++){
                        for(j = 0; j < kernel_j; j++){
                            score_error[c*kernel_i*kernel_j + i*kernel_j + j]+=kernel[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + oi*input_j+oj]*temp[i*output_j + j+oj*stride2+oi*stride1*output_j];
                        }
                    }
                }    
            }
        }

        free(temp);
    }
}

/* This function computes the errors using the backpropagation for transposed convolution
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output_error, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error){
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                kernel_error[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j] += output_error[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]*input[c*input_i*input_j + oi*input_j+oj];
                            }
                        }
                    }    
                }
            }
        }
            for(oi = 0; oi < output_i*output_j; oi++){
                bias_error[indices[s]] += output_error[indices[s]*output_i*output_j + oi];
            }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    temp[indices[s]*output_i*output_j + oi*output_j+oj] = output_error[indices[s]*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding];
                }
            }
        }
        
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                kernel_error[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j]+=temp[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]*input[c*input_i*input_j + oi*input_j+oj];
                            }
                        }
                    }    
                }
            }
        }
        
        free(temp);
    }
    * */
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    kernel_error[k][c*kernel_i*kernel_j + i*kernel_j + j]+=input[c*input_i*input_j + oi*input_j+oj]*output_error[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j];
                }
            }
        }
        if(bias_error != NULL){
            for(i = 0; i < n_kernels; i++){
                for(oi = 0; oi < output_i*output_j; oi++){
                    bias_error[i]+=output_error[i*output_i*output_j + oi];
                }
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        for(i = 0; i < n_kernels; i++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    temp[i*output_i*output_j + oi*output_j+oj] = output_error[i*(output_i-2*padding)*(output_j - 2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding];
                }
            }
        }
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    kernel_error[k][c*kernel_i*kernel_j + i*kernel_j + j]+=input[c*input_i*input_j + oi*input_j+oj]*temp[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j];
                }  
            }
        }
        if(bias_error != NULL){
            for(i = 0; i < n_kernels; i++){
                for(oi = 0; oi < output_i*output_j; oi++){
                    bias_error[i]+=temp[i*output_i*output_j + oi];
                }
            }
        }


        free(temp);
    }
}

/* This function computes the errors using the backpropagation for transposed convolution
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output_error:= the current feature map of the errors
 *                               dimensions: ((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding)
 *             @ float* input_error:= the error of the previous layer computed using the kernel and the current output_error
 *                                    dimensions: channels*input_i*input_j
 *             @ float* kernel_error:= the error of the weights computed using the input and the current output_error
 *                                    dimensions: channels*kernel_i*kernel_j
 *             @ float* bias_error:= the error of the bias
 *                                   dimensions: 1
 *             @ int stride1:= the stride used by the kernel on the feature maps of the inputs for the rows
 *             @ int stride2:= the stride used by the kernel on the feature maps of the inputs for the cols
 *             @ int padding:= the optional padding added to the output
 * */
void transposed_convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n){
    /*
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                input_error[c*input_i*input_j + oi*input_j+oj]+=output_error[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j]*kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j];
                            }
                        }
                    }    
                }
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    temp[indices[s]*output_i*output_j + oi*output_j+oj] = output_error[indices[s]*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding];
                }
            }
        }
        
        for(s = n_kernels-last_n; s < n_kernels; s++){
            for(oi = 0; oi < input_i; oi++){
                for(oj = 0; oj < input_j; oj++){
                    for(c = 0; c < channels; c++){
                        for(i = 0; i < kernel_i; i++){
                            for(j = 0; j < kernel_j; j++){
                                input_error[c*input_i*input_j + oi*input_j+oj]+=kernel[indices[s]][c*kernel_i*kernel_j + i*kernel_j + j]*temp[indices[s]*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j];
                            }
                        }
                    }    
                }
            }
        }
        
        free(temp);
    }*/
    
    int oi,oj,i,j,c,s;
    int output_i = (input_i-1)*stride1+kernel_i;
    int output_j = (input_j-1)*stride2+kernel_j;
    if(!padding){
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    input_error[c*input_i*input_j + oi*input_j+oj]+=kernel[k][c*kernel_i*kernel_j + i*kernel_j + j]*output_error[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j];

                }
            }
        }
    }
    else if(padding){
        float* temp = (float*)calloc(n_kernels*output_i*output_j,sizeof(float));
        for(i = 0; i < n_kernels; i++){
            for(oi = padding; oi < output_i-padding; oi++){
                for(oj = padding; oj < output_j-padding; oj++){
                    temp[i*output_i*output_j + oi*output_j+oj] = output_error[i*(output_i-2*padding)*(output_j-2*padding) + (oi-padding)*(output_j-2*padding)+oj-padding];
                }
            }
        }
        for(oi = 0; oi < input_i; oi++){
            for(oj = 0; oj < input_j; oj++){
                for(s = n_kernels*channels*kernel_i*kernel_j-last_n; s < n_kernels*channels*kernel_i*kernel_j; s++){
                    int k = (int)((indices[s]/(channels*kernel_i*kernel_j)));
                    i = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))/kernel_j));
                    j = ((int)(((indices[s]%(channels*kernel_i*kernel_j))%(kernel_i*kernel_j))%kernel_j));
                    c = ((int)(indices[s]%(channels*kernel_i*kernel_j))/(kernel_i*kernel_j));
                    input_error[c*input_i*input_j + oi*input_j+oj]+=kernel[k][c*kernel_i*kernel_j + i*kernel_j + j]*temp[k*output_i*output_j + i*output_j + j+oj*stride2+oi*stride1*output_j];

                } 
            }
        }

        free(temp);
    }
}

