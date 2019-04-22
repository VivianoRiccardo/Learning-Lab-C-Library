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
float sigmoid(float x){
    return 1/(1+exp(-x));
}

void sigmoid_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = sigmoid(input[i]);
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
        return x*0.1;
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
        return 0.1;
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

float derivative_mse(float y_hat, float y){
    return y_hat-y;
}

float cross_entropy(float y_hat, float y){
    float log_one;
    float constant;
    if(y_hat == 0)
        log_one = -999999;
    else
        log_one = log((double)y_hat);
    
    constant = (1-y)*(log((double)1-y_hat));
    if(constant!=constant)
        constant = 0;
    return -y*log_one-constant;
    
}


float derivative_cross_entropy(float y_hat, float y){
    return (y_hat-y)/((1-y_hat)*y_hat);
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




