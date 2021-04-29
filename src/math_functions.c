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

int min(int x, int y) {
    return (x < y) ? x : y;
}

int max(int x, int y) {
    return (x > y) ? x : y;
}


double sum_over_input(float* inputs, int dimension){
    double sum = 0;
    int i;
    for(i = 0; i < dimension; i++){
        sum+=inputs[i];
    }
    
    return sum;
}


void softmax(float* input, float* output, int size){
    int i;
    float sum = 0;
    for(i = 0; i < size; i++){
        sum+=exp(input[i]);
    }
    if(!bool_is_real(sum) || !sum){
        fprintf(stderr,"Error: not real number appeared in a softmax function!\n");
        exit(1);
    }
    for(i = 0; i < size; i++){
        output[i] = exp(input[i])/sum;
    }
}

void derivative_softmax_array(int* input, float* output,float* softmax_arr,float* error, int size){
    int i,j;
    
    for(j = 0; j < size; j++){
        if(input[j]){
            for(i = 0; i < size; i++){
                if(input[i]){
                    if (i == j)
                        output[j] += (softmax_arr[i]*(1-softmax_arr[j]))*error[i];
                    else
                        output[j] += -softmax_arr[j]*softmax_arr[i]*error[i];
                    }
            }
        }
        
    }
}
void derivative_softmax(float* output,float* softmax_arr,float* error, int size){
    int i,j;
    
    for(j = 0; j < size; j++){
        for(i = 0; i < size; i++){
            if (i == j)
                output[j] += (softmax_arr[i]*(1-softmax_arr[j]))*error[i];
            else
                output[j] -= softmax_arr[j]*softmax_arr[i]*error[i];
            
        }
        
    }
}


float sigmoid(float x){
    float t = (1+exp(-x));
    if(bool_is_real(t) && t)
    return 1/t;
    fprintf(stderr,"Error: not real number appeared in a sigmoid function!\n");
    exit(1);
}

float abs_sigmoid(float x){
    float t = (1+exp(-float_abs(x)));
    if(bool_is_real(t) && t)
    return 1/t;
    fprintf(stderr,"Error: not real number appeared in a abs_sigmoid function!\n");
    exit(1);
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
float derivative_sigmoid_given_the_sigmoid(float x){
    return x*(1-x);
}

void derivative_sigmoid_array(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_sigmoid(input[i]);
    }
}
void derivative_sigmoid_array_given_the_sigmoid(float* input, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_sigmoid_given_the_sigmoid(input[i]);
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
    if(bool_is_real(y) && (y+1))
    return (y-1)/(y+1);
    fprintf(stderr,"Error: not real number appeared in a tanh function!\n");
    exit(1);
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

float total_variation_loss_2d(float* y, int rows, int cols){
    int i,j;
    float sum;
    for(i = 0; i < rows-1; i++){
        for(j = 0; j < cols-1; j++){
            sum += float_abs(y[(i+1)*cols + j] - y[i*cols + j]) + float_abs(y[i*cols + j + 1] - y[i*cols + j]);
        }
        sum += float_abs(y[(i+1)*cols + j] - y[i*cols + j]);
    }
    for(j = 0; j < cols-1; j++){
        sum += float_abs(y[i*cols + j + 1] - y[i*cols + j]);
    }
    
    return sum;
}

void derivative_total_variation_loss_2d(float* y, float* output, int rows, int cols){
    int i,j;
    for(i = 0; i < rows-1; i++){
        for(j = 0; j < cols-1; j++){
            float ratio1 = (y[(i+1)*cols + j] - y[i*cols + j])/float_abs(y[(i+1)*cols + j] - y[i*cols + j]);
            float ratio2 = (y[i*cols + j + 1] - y[i*cols + j])/float_abs(y[i*cols + j + 1] - y[i*cols + j]);
            output[(i+1)*cols + j] += ratio1;
            output[i*cols + j] += ratio1 + ratio2;
            output[i*cols+j+1] += ratio2;
        }
        float ratio1 = (y[(i+1)*cols + j] - y[i*cols + j])/float_abs(y[(i+1)*cols + j] - y[i*cols + j]); 
        output[(i+1)*cols + j] += ratio1;
        output[i*cols + j] += ratio1;
    }
    for(j = 0; j < cols-1; j++){
        float ratio2 = (y[i*cols + j + 1] - y[i*cols + j])/float_abs(y[i*cols + j + 1] - y[i*cols + j]);
        output[i*cols + j] += ratio2;
        output[i*cols+j+1] += ratio2;
    }
    
    return;
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
    float temp,log_one;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);
    
    if(!temp || temp != temp){
        temp = 0;
        log_one = -999999;
    }
    else{
        log_one = log(temp);
    }
    return -pow((double)(1-temp),(double)gamma)*log_one;
        
}

void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = focal_loss(y_hat[i],y[i],gamma);
    }
}

float derivative_focal_loss(float y_hat, float y, float gamma){
    float temp,log_one,power;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);

    log_one = log(temp);
    power = pow((double)(1-temp),(double)gamma)/temp;

    
    float temp2 = gamma*pow((double)(1-temp),(double)gamma-1)*log_one-power;
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

void softmax_array_not_complete(float* input, float* output,int* mask, int size){
    int i;
    float sum = 0;
    for(i = 0; i < size; i++){
        if(mask[i])
            sum+=exp(input[i]);
    }
    
    for(i = 0; i < size; i++){
        if(mask[i])
        output[i] = exp(input[i])/sum;
    }
}

float elu(float z, float a){
    if (z > 0)
        return z;
    return a*(exp(z)-1);
}


void elu_array(float* input, float* output, int size, float a){
    int i;
    for(i = 0; i < size; i++){
        output[i] = elu(input[i],a);
    }
}

float derivative_elu(float z, float a){
    if (z > 0)
        return 1;
    else
        return a*exp(z);
}

void derivative_elu_array(float* input, float* output, int size, float a){
    int i;
    for(i = 0; i < size; i++){
        output[i] = derivative_elu(input[i],a);
    }
}

/* This function computes the dot product between 2 array, input and input2
 * with the same length, and store the result in the output array
 * 
 * Input:
 * 
 *             @ float* input1:= the first input array
 *             @ float* input2:= the second input array
 *             @ float* output:= the output array
 *             @ int size:= the size of input1, input2, input3
 * */
void dot1D(float* input1, float* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]*input2[i];
    }
}

/* This function computes the sum between 2 array, input and input2
 * with the same length, and store the result in the output array
 * 
 * Input:
 * 
 *             @ float* input1:= the first input array
 *             @ float* input2:= the second input array
 *             @ float* output:= the output array
 *             @ int size:= the size of input1, input2, input3
 * */
void sum1D(float* input1, float* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]+input2[i];
    }
}


/* This function computes a dot product between an array and a float value: value
 * 
 * Input
 * 
 *             @ float* input:= the imput used to compute the output
 *             @ float value:= the float value that must be multiplied for the inputs
 *             @ float* output:= the array where you need to store the output
 *             @ int dimension:= the dimension of input and output
 * 
 * */
void mul_value(float* input, float value, float* output, int dimension){
    int i;
    for(i = 0; i < dimension; i++){
        output[i] = input[i]*value;
    }
}

/* This function sum the partial derivatives of the residual layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(exists_d_kernels_cl(m->rls[i]->cls[j]) || exists_edge_popup_stuff_cl(m->rls[i]->cls[j])){
                if(exists_d_kernels_cl(m->rls[i]->cls[j])){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                        sum1D(m->rls[i]->cls[j]->d_kernels[k],m2->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->channels*m3->rls[i]->cls[j]->kernel_rows*m3->rls[i]->cls[j]->kernel_cols);
                    }
                }
                
                if(exists_d_biases_cl(m->rls[i]->cls[j]))
                sum1D(m->rls[i]->cls[j]->d_biases,m2->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->n_kernels);
                if(exists_edge_popup_stuff_cl(m->rls[i]->cls[j]))
                sum1D(m->rls[i]->cls[j]->d_scores,m2->rls[i]->cls[j]->d_scores,m3->rls[i]->cls[j]->d_scores,m3->rls[i]->cls[j]->n_kernels);

                if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                    }
                }
            }
        }
    }
}



/* This function sum the partial derivatives of the convolutional layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(exists_d_kernels_cl(m->cls[j])){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                sum1D(m->cls[j]->d_kernels[k],m2->cls[j]->d_kernels[k],m3->cls[j]->d_kernels[k],m3->cls[j]->channels*m3->cls[j]->kernel_rows*m3->cls[j]->kernel_cols);
            }
            
            sum1D(m->cls[j]->d_biases,m2->cls[j]->d_biases,m3->cls[j]->d_biases,m3->cls[j]->n_kernels);
            if(exists_edge_popup_stuff_with_only_training_mode_cl(m->cls[j]))
            sum1D(m->cls[j]->d_scores,m2->cls[j]->d_scores,m3->cls[j]->d_scores,m3->cls[j]->n_kernels);

            if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                }
            }
        }
    }

}



/* This function sum the partial derivatives of the fully-connected layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(exists_d_params_fcl(m->fcls[i])){
            sum1D(m->fcls[i]->d_weights,m2->fcls[i]->d_weights,m3->fcls[i]->d_weights,m->fcls[i]->input*m->fcls[i]->output);    
            sum1D(m->fcls[i]->d_biases,m2->fcls[i]->d_biases,m3->fcls[i]->d_biases,m->fcls[i]->output);
        }
        if(exists_edge_popup_stuff_fcl(m->fcls[i])){
            sum1D(m->fcls[i]->d_scores,m2->fcls[i]->d_scores,m3->fcls[i]->d_scores,m->fcls[i]->output*m->fcls[i]->input);   
        } 
        if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
            sum1D(m->fcls[i]->layer_norm->d_gamma,m2->fcls[i]->layer_norm->d_gamma,m3->fcls[i]->layer_norm->d_gamma,m->fcls[i]->layer_norm->vector_dim);
    }
    
        
}




/* This function sum the partial derivatives of the lstm layers of a rmodel m and a second rmodel m2 in a third rmodel m3
 * 
 * Input:
 * 
 *             @ rmodel* m:= the input rmodel
 *             @ rmodel* m2:= another input rmodel
 *             @ rmodel* m3:= the output rmodel
 * 
 * */
void sum_lstm_layers_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            if(m->lstms[i]->training_mode == GRADIENT_DESCENT || m->lstms[i]->training_mode == FREEZE_TRAINING){
                sum1D(m->lstms[i]->d_w[j],m2->lstms[i]->d_w[j],m3->lstms[i]->d_w[j],m->lstms[i]->output_size*m->lstms[i]->input_size);
                sum1D(m->lstms[i]->d_u[j],m2->lstms[i]->d_u[j],m3->lstms[i]->d_u[j],m->lstms[i]->output_size*m->lstms[i]->output_size);
                sum1D(m->lstms[i]->d_biases[j],m2->lstms[i]->d_biases[j],m3->lstms[i]->d_biases[j],m->lstms[i]->output_size);
            }
            
            else if(m->lstms[i]->training_mode == EDGE_POPUP){
                sum1D(m->lstms[i]->d_w_scores[j],m2->lstms[i]->d_w_scores[j],m3->lstms[i]->d_w_scores[j],m->lstms[i]->output_size*m->lstms[i]->input_size);
                sum1D(m->lstms[i]->d_u_scores[j],m2->lstms[i]->d_u_scores[j],m3->lstms[i]->d_u_scores[j],m->lstms[i]->output_size*m->lstms[i]->output_size);
            }
        }
    
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell; j++){
                sum1D(m->lstms[i]->bns[j]->d_gamma,m2->lstms[i]->bns[j]->d_gamma,m3->lstms[i]->bns[j]->d_gamma,m->lstms[i]->bns[j]->vector_dim);
                sum1D(m->lstms[i]->bns[j]->d_beta,m2->lstms[i]->bns[j]->d_beta,m3->lstms[i]->bns[j]->d_beta,m->lstms[i]->bns[j]->vector_dim);
            }
        }
    }
    

}

/* the absolute value of a float number*/
float float_abs(float a){
    return (a > 0) ? a : -a;
}

/* absolute value of each value of an array*/
void float_abs_array(float* a, int n){
    int i;
    for(i = 0; i < n; i++){
        a[i] = float_abs(a[i]);
    }
}


/* absolute value of each value of an array*/
float* get_float_abs_array(float* a, int n){
    float* m = (float*)calloc(n,sizeof(float));
    int i;
    for(i = 0; i < n; i++){
        m[i] = float_abs(a[i]);
    }
    return m;
}

void dot_float_input(float* input1, int* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = (float)(input1[i]*input2[i]);
    }
}

/* This function sum the partial derivatives in model m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ model* m:= first input model
 *             @ model* m2:= second input model
 *             @ model* m3:= output model
 * 
 * */
void sum_model_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_fully_connected_layers_partial_derivatives(m,m2,m3);
    sum_convolutional_layers_partial_derivatives(m,m2,m3);
    sum_residual_layers_partial_derivatives(m,m2,m3);
}

/*sum partial derivatives of batch sizes in 1 unique model
 * 
 * input:
 * 
 *             @ model* sum_m:= where are summed up the partial derivatives
 *             @ model** models:= the models (dimension: n_models)
 *             @ int n_models:= the number of models
 * 
 * */
void sum_models_partial_derivatives(model* sum_m, model** models, int n_models){
    int i;
    for(i = 0; i < n_models; i++){
        sum_model_partial_derivatives(models[i],sum_m,sum_m);
    }
}

/* This function sum the partial derivatives in rmodel m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ rmodel* m:= first input rmodel
 *             @ rmodel* m2:= second input rmodel
 *             @ rmodel* m3:= output rmodel
 * 
 * */
void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_lstm_layers_partial_derivatives(m,m2,m3);
}

/* This function sum the partial derivatives in rmodel m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ rmodel* m:= first input rmodel
 *             @ rmodel* m2:= second input rmodel
 *             @ rmodel* m3:= output rmodel
 * 
 * */
void sum_rmodels_partial_derivatives(rmodel* m, rmodel** m2, int n_models){
    if(m == NULL || m2 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    int i;
    for(i = 0; i < n_models; i++){
        sum_rmodel_partial_derivatives(m,m2[i],m);
    }
}

void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3){
    if(vm == NULL || vm2 == NULL || vm3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_vae_model_partial_derivatives\n");
        exit(1);
    }
    sum_model_partial_derivatives(vm->encoder,vm2->encoder,vm3->encoder);
    sum_model_partial_derivatives(vm->decoder,vm2->decoder,vm3->decoder);
}

