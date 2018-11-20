#include "llab.h"
 
 /* This function compute the local response normalization for a convolutional layer
  * 
  * Input:
  *           @ float* tensor:= is the tensor of feature map of the convolutional layer
  *                                 dimensions: tensor_depth*tensor_i*tensor_j
  *           @ float* output:= is the tensor of the output, or is the "tensor" normalized
  *                                 dimensions: tensor_depth*tensor_i*tensor_j
  *           @ int index_ac:= is the channel where the "single input" that must be normalized is
  *           @ int index_ai:= is the row where the "single input" that must be normalized is
  *           @ int index_aj:= is the column where the "single input" that must be normalized is
  *           @ int tensor_depth:= is the number of the channels of tensor and output
  *           @ int tensor_i:= is the number of rows of each feature map of tensor and output
  *           @ int tensor_j:= is the number of columns of each feature map of tensor and output
  *           @ float n_constant:= is an hyper parameter (usually 5)
  *           @ float beta:= is an hyper parameter (usually 0.75)
  *           @ float alpha:= is an hyper parameter (usually 0.0001)
  *           @ float k:= is an hyper parameter(usually 2)
  * */
void local_response_normalization_feed_forward(float* tensor,float* output, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k){
    int i,j,c;
    int lower_bound,upper_bound;
    float sum = 0;
    float temp;
    if(index_ac-(int)(n_constant/2) < 0)
        lower_bound = 0;
    else
        lower_bound = index_ac-(int)(n_constant/2);
        
    if(index_ac+(int)(n_constant/2) > tensor_depth-1)
        upper_bound = tensor_depth-1;
    else
        upper_bound = index_ac+(int)(n_constant/2);
    
    for(c = lower_bound; c <= upper_bound; c++){
        temp = tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj];
        sum += temp*temp;
    }
    sum = k+alpha*sum;
    sum = (float)pow((double)sum,(double)beta);
    output[index_ac*tensor_i*tensor_j + index_ai*tensor_j + index_aj] = tensor[index_ac*tensor_i*tensor_j + index_ai*tensor_j + index_aj]/sum;
}

 /* This function compute the local response normalization for a convolutional layer
  * 
  * Input:
  *           @ float* tensor:= is the tensor of feature map of the convolutional layer
  *                                 dimensions: tensor_depth*tensor_i*tensor_j
  *           @ float* tensor_error:= is the error of the tensor of feature map of the convolutional layer
  *                                 dimensions: tensor_depth*tensor_i*tensor_j
  *           @ float* output_error:= is the tensor of the error of the output
  *                                 dimensions: tensor_depth*tensor_i*tensor_j
  *           @ int index_ac:= is the channel where the "single input" has been normalized is
  *           @ int index_ai:= is the row where the "single input" has been normalized is
  *           @ int index_aj:= is the column where the "single input" has been normalized is
  *           @ int tensor_depth:= is the number of the channels of tensor and output
  *           @ int tensor_i:= is the number of rows of each feature map of tensor and output
  *           @ int tensor_j:= is the number of columns of each feature map of tensor and output
  *           @ float n_constant:= is an hyper parameter (usually 5)
  *           @ float beta:= is an hyper parameter (usually 0.75)
  *           @ float alpha:= is an hyper parameter (usually 0.0001)
  *           @ float k:= is an hyper parameter(usually 2)
  * */
void local_response_normalization_back_prop(float* tensor,float* tensor_error,float* output_error, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k){
    int i,j,c;
    int lower_bound, upper_bound;
    float sum = 0;
    float temp;
    if(index_ac-(int)(n_constant/2) < 0)
        lower_bound = 0;
    else
        lower_bound = index_ac-(int)(n_constant/2);
        
    if(index_ac+(int)(n_constant/2) > tensor_depth-1)
        upper_bound = tensor_depth-1;
    else
        upper_bound = index_ac+(int)(n_constant/2);
    
    for(c = lower_bound; c <= upper_bound; c++){
        temp = tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj];
        sum += temp*temp;
    }
    
    sum = k+alpha*sum;
    temp = sum;
    sum = (float)pow((double)sum,(double)beta);
    temp = (float)pow((double)temp,(double)beta+1);
    
    for(c = lower_bound; c <= upper_bound; c++){
        if(c == index_ac)
            tensor_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj] += ((float)(1/sum)-(float)(2*beta*alpha*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj])/temp);
        
        else
            tensor_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj] += (-(float)(2*beta*alpha*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*tensor[index_ac*tensor_i*tensor_j + index_ai*tensor_j + index_aj])/temp);
    }
}
