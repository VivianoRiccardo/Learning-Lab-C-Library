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


/* This computes the batch normalization across batches
 * 
 * Input:
 * 
 *             @ int batch_size:= the size of the batch (number of total instances actually running)
 *             @ float** input_vectors:= the total instances running, dimensions: batch_size*size_vectors
 *             @ float** temp_vectors:= a temporary vector where we store the h_hat_i, dimensions:= batch_size*size_vectors
 *             @ int size_vectors:= the size of each vector
 *             @ float* gamma:= the parameters that we must learn
 *             @ float* beta:= other params that we must learn
 *             @ float* mean:= a vector initialized with all 0s where we store the mean
 *             @ float* var:= a vector initialized with all 0s where we store the variance
 *             @ float** outputs:= where we store the outputs coming from this normalization
 *             @ float epsilon:= a param that let us to avoid division by 0
 * 
 * */
void batch_normalization_feed_forward(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon){
    int i,j;
    float temp;
    /*mean*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            mean[j] += input_vectors[i][j];
            if(i == batch_size-1)
                mean[j]/=(float)batch_size;
        }
    }
    
    /*variance*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            temp = input_vectors[i][j]-mean[j];
            temp = temp*temp;
            var[j] += temp;
            if(i == batch_size-1)
                var[j]/=(float)batch_size;
        }
    }
    
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            temp_vectors[i][j] = (input_vectors[i][j]-mean[j])/(sqrtf(var[j]+epsilon));
            outputs[i][j] = temp_vectors[i][j]*gamma[j] + beta[j];
        }
    }

}

/* This Function computes the error from a batch normalization
 * 
 * Input:
 * 
 *             @ int batch_size:= the size of the batch (number of total instances actually running)
 *             @ float** input_vectors:= the total instances running, dimensions: batch_size*size_vectors
 *             @ float** temp_vectors:= a temporary vector where we store the h_hat_i, dimensions:= batch_size*size_vectors
 *             @ int size_vectors:= the size of each vector
 *             @ float* gamma:= the parameters that we must learn
 *             @ float* beta:= other params that we must learn
 *             @ float* mean:= a vector initialized with all 0s where we store the mean
 *             @ float* var:= a vector initialized with all 0s where we store the variance
 *             @ float** outputs_error:= where are stored the output errors coming from the next layer
 *             @ float* gamma_error:= where we store the partial derivatives of gamma
 *             @ float* beta_error:= where we store the partial derivatives of beta
 *             @ float** input_error:= where we store the input error
 *             @ float** temp_vectors_error:= useful for the computation
 *             @ float* temp_array:= useful for the computation
 *             @ float epsilon:= a param that let us to avoid division by 0
 * 
 * */
void batch_normalization_back_prop(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon){
    int i,j,z;
    
    for(i = 0; i < size_vectors; i++){
        mean[i] = 0;
    }
    /* gamma and beta error*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            gamma_error[j] += outputs_error[i][j]*temp_vectors[i][j];
            beta_error[j] += outputs_error[i][j];
            temp_vectors_error[i][j] = outputs_error[i][j]*gamma[j];
            temp_array[j] += temp_vectors_error[i][j];
            mean[j] += temp_vectors_error[i][j]*temp_vectors[i][j];
        }
    }
    
    for(j = 0; j < batch_size; j++){
        for(z = 0; z < size_vectors; z++){
            
            input_error[j][z] += (batch_size*temp_vectors_error[j][z] - temp_array[j] - temp_vectors[j][z]*mean[j])/(batch_size*sqrtf(var[j]+epsilon));
        }
    }

}


/* This function computes the final mean and variance for a bn layer once the training is ended, according to the 
 * second part of the pseudocode that you can find here: https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/
 * 
 * Input:
 *     
 *             @ float** input_vectors:= the input that comes just before of this bn* layer, coming from all the instances of the training,
 *                                       dimensions: n_vectors x vector_size
 *            @ int n_vectors:= the first dimension of input_vectors
 *             @ int vector_size:= the second dimension of ninput_vectors
 *             @ int mini_batch_size:= the batch size used during the training
 *             @ bn* bn_layer:= the batch normalized layer where the final mean and final variance will be set up
 * 
 * */
void batch_normalization_final_mean_variance(float** input_vectors, int n_vectors, int vector_size, int mini_batch_size, bn* bn_layer){
    int i,j;
    float* mean = (float*)calloc(vector_size,sizeof(float));
    float* var = (float*)calloc(vector_size,sizeof(float));
    srand(time(NULL));
    shuffle_float_matrix(input_vectors, n_vectors);
    
    if(n_vectors%mini_batch_size != 0){
        fprintf(stderr,"Error: your batch_size doesn't divide your n_vectors perfectly\n");
        exit(1);
    }
    for(i = 0; i < n_vectors; i+=mini_batch_size){
        reset_bn(bn_layer);
        batch_normalization_feed_forward(mini_batch_size,input_vectors,bn_layer->temp_vectors,vector_size,bn_layer->gamma,bn_layer->beta,bn_layer->mean,bn_layer->var, bn_layer->outputs,EPSILON);
        sum1D(bn_layer->mean,mean,mean,vector_size);
        sum1D(bn_layer->var,var,var,vector_size);
        
    }
    
    for(i = 0; i < vector_size; i++){
        mean[i] /= (float)(n_vectors/mini_batch_size);
        var[i] = (float)((float)mini_batch_size/(float)(mini_batch_size-1))*var[i]/(float)(n_vectors/mini_batch_size);
    }
    
    copy_array(mean,bn_layer->final_mean,vector_size);
    copy_array(var,bn_layer->final_var,vector_size);
    
    free(mean);
    free(var);
    
    return;
}

/* This function computes the group normalization feed forward for a convolutional layer
 * 
 * Inputs:
 * 
 *                 @ float* tensor:= the input of the convolutional layer
 *                 @ int tensor_c:= the number of channels of the tensor
 *                 @ int tensor_i:= the number of rows of the tensor
 *                 @ int tensor_j:= the number of columns of the tensor
 *                 @ int n_channels:= the grouped channels for the group normalization
 *                 @ int stride:= the stride between channels for the group normalization
 *                 @ bn** bns:= the bns layer where is gonne be computed the group normalization
 *                 @ int pad_i:= the padding for the rows of the tensor
 *                 @ int pad_j:= tha padding for the columns of the tensor
 *                 @ float* post_normalization:= where the post normalization output is stored, size = tensor_c*tensor_i*tensor_j
 * 
 * */
void group_normalization_feed_forward(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, int pad_i, int pad_j, float* post_normalization){
    int n_bns = (tensor_c-n_channels)/stride+1;
    int i,j,k,rows,cols;
    
    float** input_vector = (float**)malloc(sizeof(float*)*n_channels);
    for(i = 0; i < n_channels; i++){
        input_vector[i] = (float*)calloc(tensor_i*tensor_j - 2*pad_i - 2*pad_j,sizeof(float));
    }
    
    for(i = 0, j = 0; j < n_bns; i+=stride,j++){
        for(k = i; k < i+stride; k++){
            for(rows = pad_i; rows < tensor_i-pad_i; rows++){
                for(cols = pad_j; cols < tensor_j-pad_j; cols++){
                    input_vector[k-i][(rows-pad_i)*(tensor_j-2*pad_j)+cols-pad_j] = tensor[k*tensor_i*tensor_j+rows*tensor_j+cols];
                }
            }
        }
        batch_normalization_feed_forward(n_channels,input_vector,bns[j]->temp_vectors, bns[j]->vector_dim, bns[j]->gamma, bns[j]->beta, bns[j]->mean, bns[j]->var, bns[j]->outputs,bns[j]->epsilon);
        for(k = i; k < i+stride; k++){
            for(rows = pad_i; rows < tensor_i-pad_i; rows++){
                for(cols = pad_j; cols < tensor_j-pad_j; cols++){
                    post_normalization[k*tensor_i*tensor_j+rows*tensor_j+cols] = bns[j]->outputs[k-i][(rows-pad_i)*(tensor_j-2*pad_j)+cols-pad_j];
                }
            }
        }
    }
    
    for(i = 0; i < n_channels; i++){
        free(input_vector[i]);
    }
    
    free(input_vector);
} 


/* This function computes the group normalization back propagation for a convolutional layer
 * 
 * Inputs:
 * 
 *                 @ float* tensor:= the input of the convolutional layer
 *                 @ int tensor_c:= the number of channels of the tensor
 *                 @ int tensor_i:= the number of rows of the tensor
 *                 @ int tensor_j:= the number of columns of the tensor
 *                 @ int n_channels:= the grouped channels for the group normalization
 *                 @ int stride:= the stride between channels for the group normalization
 *                 @ bn** bns:= the bns layer where is gonne be computed the group normalization
 *                 @ float* error:= tensor_c*tensor_i*tensor_j
 *                 @ int pad_i:= the padding for the rows of the tensor
 *                 @ int pad_j:= tha padding for the columns of the tensor
 *                 @ float* input_error:= where is stored the error of the input, size = tensor_c*tensor_i*tensor_j
 * 
 * */
void group_normalization_back_propagation(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, float* ret_error,int pad_i, int pad_j, float* input_error){
    int n_bns = (tensor_c-n_channels)/stride+1;
    int i,j,k,rows,cols;
    
    float** error = (float**)malloc(sizeof(float*)*n_channels);
    float** input_vector = (float**)malloc(sizeof(float*)*n_channels);
    
    for(i = 0; i < n_channels; i++){
        error[i] = (float*)calloc(tensor_i*tensor_j-2*pad_i-2*pad_j,sizeof(float));
        input_vector[i] = (float*)calloc(tensor_i*tensor_j-2*pad_i-2*pad_j,sizeof(float));
    }
    
    
    for(i = 0, j = 0; j < n_bns; i+=stride,j++){
        for(k = i; k < i+stride; k++){
            for(rows = pad_i; rows < tensor_i-pad_i; rows++){
                for(cols = pad_j; cols < tensor_j-pad_j; cols++){
                    input_vector[k-i][(rows-pad_i)*(tensor_j-2*pad_j)+cols-pad_j] = tensor[k*tensor_i*tensor_j+rows*tensor_j+cols];
                    error[k-i][(rows-pad_i)*(tensor_j-2*pad_j)+cols-pad_j] = ret_error[k*tensor_i*tensor_j+rows*tensor_j+cols];
                }
            }
        }
        batch_normalization_back_prop(n_channels,input_vector,bns[j]->temp_vectors, bns[j]->vector_dim, bns[j]->gamma, bns[j]->beta, bns[j]->mean, bns[j]->var,error,bns[j]->d_gamma, bns[j]->d_beta,bns[j]->error2, bns[j]->temp1,bns[j]->temp2, bns[j]->epsilon);
        for(k = i; k < i+stride; k++){
            for(rows = pad_i; rows < tensor_i-pad_i; rows++){
                for(cols = pad_j; cols < tensor_j-pad_j; cols++){
                    input_error[k*tensor_i*tensor_j+rows*tensor_j+cols] = bns[j]->error2[k-i][(rows-pad_i)*(tensor_j-2*pad_j)+cols-pad_j];
                }
            }
        }
    
    }
    
    for(i = 0; i < n_channels; i++){
        free(input_vector[i]);
        free(error[i]);
    }
    
    free(input_vector);
    free(error);
}
