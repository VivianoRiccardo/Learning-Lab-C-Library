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

void softmax(float* input, float* output, int size){
    int i;
    float sum = 0;
    for(i = 0; i < size; i++){
        sum+=exp(input[i]);
    }
    
    for(i = 0; i < size; i++){
        output[i] = exp(input[i])/sum;
    }
}

void derivative_softmax_array(float* input, float* output,float* softmax_arr,float* error, int size){
    int i,j;
    
    for(j = 0; j < size; j++){
        for(i = 0; i < size; i++){
            if (i == j)
                output[j] += (softmax_arr[i]*(1-softmax_arr[j]))*error[i];
            else
                output[j] += -softmax_arr[j]*softmax_arr[i]*error[i];
        }
        
    }
}


float sigmoid(float x){
    return 1/(1+exp(-x));
}

float abs_sigmoid(float x){
    return 1/(1+exp(-float_abs(x)));
}
void sigmoid_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = sigmoid(input[i]);
    }
}

void abs_sigmoid_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = abs_sigmoid(input[i]);
    }
}

float derivative_sigmoid(float x){
    float y = sigmoid(x);
    return y*(1-y);
}

void derivative_sigmoid_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_sigmoid(input[i]);
    }
}

float relu(float x){
    if(x > 0)
        return x;
    else
        return 0;
}

void relu_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = relu(input[i]);
    }
}

float derivative_relu(float x){
    if(x > 0)
        return 1;
    else
        return 0;
}

void derivative_relu_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_relu(input[i]);
    }
}

float leaky_relu(float x){
    if(x > 0)
        return x;
    else
        return x*LEAKY_RELU_THRESHOLD;
}

void leaky_relu_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = leaky_relu(input[i]);
    }
}

float derivative_leaky_relu(float x){
    if(x > 0)
        return 1;
    else
        return LEAKY_RELU_THRESHOLD;
}

void derivative_leaky_relu_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_leaky_relu(input[i]);
    }
}

float tanhh(float x){
    float y = exp(2*x);
    return (y-1)/(y+1);
}

void tanhh_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = tanhh(input[i]);
    }
}

float derivative_tanhh(float x){
    float y = tanhh(x);
    return 1-y*y;
}

void derivative_tanhh_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_tanhh(input[i]);
    }
}

float mse(float y_hat, float y){
    float z = y_hat-y;
    return z*z/2;
}

void mse_array(float* y_hat, float* y, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = mse(y_hat[i],y[i]);
    }
}

float derivative_mse(float y_hat, float y){
    return y_hat-y;
}

void derivative_mse_array(float* y_hat, float* y, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_mse(y_hat[i],y[i]);
    }
}

float cross_entropy(float y_hat, float y){
    float log_one;
    float constant;
    if(y_hat == 0 || y_hat != y_hat)
        log_one = -999999;
    else
        log_one = log((double)y_hat);
    
    constant = (1-y)*(log((double)1-y_hat));
    if(constant!=constant)
        constant = 0;
    return -y*log_one-constant;
    
}

void cross_entropy_array(float* y_hat, float* y, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = cross_entropy(y_hat[i],y[i]);
    }
}


float derivative_cross_entropy(float y_hat, float y){
    return (y_hat-y)/((1-y_hat)*y_hat);
}

void derivative_cross_entropy_array(float* y_hat, float* y, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_cross_entropy(y_hat[i],y[i]);
    }
}

float cross_entropy_reduced_form(float y_hat, float y){
    float log_one;
    if(y_hat == 0)
        log_one = -999999;
    else
        log_one = log((double)y_hat);
    
    return -y*log_one;
    
}


float derivative_cross_entropy_reduced_form_with_softmax(float y_hat, float y){
    return y_hat -y;
}

void derivative_cross_entropy_reduced_form_with_softmax_array(float* y_hat, float* y,float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_cross_entropy_reduced_form_with_softmax(y_hat[i],y[i]);
    }
}

float huber_loss(float y_hat, float y, float threshold){
    if(y_hat >= y){
        if((y_hat - y) <= threshold)
            return (y_hat-y)*(y_hat-y)/2;
        else
            return threshold*(y_hat-y)-threshold*threshold/2;
    }
    else{
        if((y - y_hat) <= threshold)
            return (y-y_hat)*(y-y_hat)/2;
        else
            return threshold*(y-y_hat)-threshold*threshold/2;    
    }
}

float derivative_huber_loss(float y_hat, float y, float threshold){
    if(y_hat >= y){
        if((y_hat - y) <= threshold)
            return (y_hat-y);
        else
            return threshold;
    }
    else{
        if((y - y_hat) <= threshold)
            return (y-y_hat);
        else
            return -threshold;    
    }
}

void derivative_huber_loss_array(float* y_hat, float* y,float* output, float threshold, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_huber_loss(y_hat[i],y[i],threshold);
    }
}

/*used for a tanh*/
float modified_huber_loss(float y_hat, float y, float threshold1, float threshold2){
    if(y*y_hat>= -1){
        if(threshold1-y_hat*y > 0)
            return (threshold1-y_hat*y)*(threshold1-y_hat*y);
        else
            return 0;
    }
    
    else{
        return -threshold2*y_hat*y;
    }
}

float derivative_modified_huber_loss(float y_hat, float y, float threshold1, float threshold2){
    if(y*y_hat>= -1){
        if(threshold1-y_hat*y > 0)
            return -2*y*threshold1+2*y_hat*y*y;
        else
            return 0;
    }
    
    else{
        return -threshold2*y;
    }
}

void derivative_modified_huber_loss_array(float* y_hat, float* y, float threshold1, float* output, float threshold2, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_modified_huber_loss(y_hat[i],y[i],threshold1,threshold2);
    }
}


float focal_loss(float y_hat, float y, float gamma){
    float temp;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);
    
    return -pow((double)(1-temp),(double)gamma)*log(temp);
        
}

void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = focal_loss(y_hat[i],y[i],gamma);
    }
}

float derivative_focal_loss(float y_hat, float y, float gamma){
    float temp;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);
    float temp2 = gamma*pow((double)(1-temp),(double)gamma-1)*log(temp)-pow((double)(1-temp),(double)gamma)/temp;
    if(y == 1)
        return temp2;
    else
        return -temp2;
}

void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_focal_loss(y_hat[i],y[i],gamma);
    }
}

void kl_divergence(float* input1, float* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]*log((double)(input1[i]/input2[i]));
    }
}

void derivative_kl_divergence(float* y_hat, float* y, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = log((double)(y_hat[i]/y[i]))+1;
    }
}

float entropy(float y_hat){
    return -y_hat*log((double)y_hat);
}

void entropy_array(float* y_hat, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = entropy(y_hat[i]);
    }
}

float derivative_entropy(float y_hat){
    return -1-log((double)y_hat);
}

void derivative_entropy_array(float* y_hat, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_entropy(y_hat[i]);
    }
}


