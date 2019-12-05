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
            //tensor_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj] += ((float)(1/sum)-(float)(2*beta*alpha*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj])/temp);
            tensor_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj] += output_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*((float)(1/sum)-(float)(2*beta*alpha*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj])/temp);
        
        else
            tensor_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj] += output_error[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*(-(float)(2*beta*alpha*tensor[c*tensor_i*tensor_j + index_ai*tensor_j + index_aj]*tensor[index_ac*tensor_i*tensor_j + index_ai*tensor_j + index_aj])/temp);
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

/* This computes the mean and variance for the batch normalization across batches
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
void batch_normalization_feed_forward_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon){
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

}

/* This computes the batch norm for the batch normalization across batches
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
void batch_normalization_feed_forward_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon, int i){
	int j;
    for(j = 0; j < size_vectors; j++){
        temp_vectors[i][j] = (input_vectors[i][j]-mean[j])/(sqrtf(var[j]+epsilon));
        outputs[i][j] = temp_vectors[i][j]*gamma[j] + beta[j];
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


    /* gamma and beta error*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            gamma_error[j] += outputs_error[i][j]*temp_vectors[i][j];
            beta_error[j] += outputs_error[i][j];
            temp_vectors_error[i][j] = outputs_error[i][j]*gamma[j];
            temp_array[j] += input_vectors[i][j] - mean[j];
        }
    }

    /* input_error*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < batch_size; j++){
            for(z = 0; z < size_vectors; z++){
                if(i == j)
                    input_error[j][z] += temp_vectors_error[j][z]*((float)(1-(float)1/batch_size)/(float)sqrtf(var[z]+epsilon)-(float)((input_vectors[j][z]-mean[z])*2*(float)(1-(float)1/batch_size)*(input_vectors[j][z]-mean[z])-(float)(2/batch_size)*(temp_array[z]-input_vectors[j][z]+mean[z]))/(float)((2*batch_size)*(pow((double)var[z]+epsilon,3/2))));
                else
                    input_error[j][z] += temp_vectors_error[j][z]*(-(float)(sqrtf((float)var[z]+epsilon)/(float)batch_size)-(float)((input_vectors[i][z]-mean[z])*2*(float)(1-(float)1/(float)batch_size)*(float)(input_vectors[j][z]-mean[z])-((float)2/(float)batch_size)*(float)(temp_array[z]-input_vectors[j][z]+mean[z]))/(float)((2*batch_size)*(pow((double)var[z]+epsilon,3/2))));

            }
        }
    }

}

/* This Function computes the first step for the error from a batch normalization
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
void batch_normalization_back_prop_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon){
    int i,j,z;


    /* gamma and beta error*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < size_vectors; j++){
            gamma_error[j] += outputs_error[i][j]*temp_vectors[i][j];
            beta_error[j] += outputs_error[i][j];
            temp_vectors_error[i][j] = outputs_error[i][j]*gamma[j];
            temp_array[j] += input_vectors[i][j] - mean[j];
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
void batch_normalization_back_prop_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon, int j){
    int i,z;

    /* input_error*/
    for(i = 0; i < batch_size; i++){    
        for(z = 0; z < size_vectors; z++){
            if(i == j)
                input_error[j][z] += temp_vectors_error[j][z]*((float)(1-(float)1/batch_size)/(float)sqrtf(var[z]+epsilon)-(float)((input_vectors[j][z]-mean[z])*2*(float)(1-(float)1/batch_size)*(input_vectors[j][z]-mean[z])-(float)(2/batch_size)*(temp_array[z]-input_vectors[j][z]+mean[z]))/(float)((2*batch_size)*(pow((double)var[z]+epsilon,3/2))));
            else
                input_error[j][z] += temp_vectors_error[j][z]*(-(float)(sqrtf((float)var[z]+epsilon)/(float)batch_size)-(float)((input_vectors[i][z]-mean[z])*2*(float)(1-(float)1/(float)batch_size)*(float)(input_vectors[j][z]-mean[z])-((float)2/(float)batch_size)*(float)(temp_array[z]-input_vectors[j][z]+mean[z]))/(float)((2*batch_size)*(pow((double)var[z]+epsilon,3/2))));

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

/* same of batch normalization feed forward but with 1d arrays and padding rows*/
void channel_normalization_feed_forward(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs,float epsilon, int rows_pad, int cols_pad, int rows, int cols){
    int i,j,k;
    float temp;
    /*mean*/
    for(i = 0; i < batch_size; i++){
        for(j = rows_pad; j < rows-rows_pad; j++){
            for(k = cols_pad; k < cols-cols_pad; k++){
                mean[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] += input_vectors[i*rows*cols+j*cols+k];
                if(i == batch_size-1)
                    mean[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad]/=(float)batch_size;    
            }
            
        }
    }
    
    /*variance*/
    for(i = 0; i < batch_size; i++){
        for(j = rows_pad; j < rows-rows_pad; j++){
            for(k = cols_pad; k < cols-cols_pad; k++){
                temp = input_vectors[i*rows*cols+j*cols+k]-mean[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad];
                temp = temp*temp;
                var[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] += temp;
                if(i == batch_size-1)
                    var[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad]/=(float)batch_size;
            }
        }
    }
    
    for(i = 0; i < batch_size; i++){
        for(j = rows_pad; j < rows-rows_pad; j++){
            for(k = cols_pad; k < cols-cols_pad; k++){
                temp_vectors[i][(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] = (input_vectors[i*rows*cols+j*cols+k]-mean[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad])/(sqrtf(var[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad]+epsilon));
                outputs[i*rows*cols+j*cols+k] = temp_vectors[i][(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad]*gamma[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] + beta[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad];
            }
        }
    }

}

/* same of batch norm back prop with arrays and padding*/
void channel_normalization_back_prop(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs_error, float* gamma_error, float* beta_error, float* input_error, float** temp_vectors_error,float* temp_array, float epsilon, int rows_pad, int cols_pad, int rows, int cols){
    int i,j,k,z;


    /* gamma and beta error*/
    for(i = 0; i < batch_size; i++){
        for(j = rows_pad; j < rows-rows_pad; j++){
            for(k = cols_pad; k < cols-cols_pad; k++){
                gamma_error[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] += outputs_error[i*rows*cols+j*cols+k]*temp_vectors[i][(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad];
                beta_error[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] += outputs_error[i*rows*cols+j*cols+k];
                temp_vectors_error[i][(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] = outputs_error[i*rows*cols+j*cols+k]*gamma[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad];
                temp_array[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad] += input_vectors[i*rows*cols+j*cols+k] - mean[(j-rows_pad)*(cols-2*cols_pad)+k-cols_pad];
            }
        }
    }

    /* input_error*/
    for(i = 0; i < batch_size; i++){
        for(j = 0; j < batch_size; j++){
            for(z = rows_pad; z < rows-rows_pad; z++){
                for(k = cols_pad; k < cols-cols_pad; k++){
                if(i == j)
                    input_error[j*rows*cols+z*cols+k] += temp_vectors_error[j][(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]*(float)((float)(1-(float)1/batch_size)/sqrtf(var[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]+epsilon)-(float)((input_vectors[j*rows*cols+z*cols+k]-mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad])*2*(float)(1-(float)1/batch_size)*(input_vectors[j*rows*cols+z*cols+k]-mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad])-(float)(2/batch_size)*(temp_array[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]-input_vectors[j*rows*cols+z*cols+k]+mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]))/(float)((2*batch_size)*(pow((double)var[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]+epsilon,3/2))));
                else
                    input_error[j*rows*cols+z*cols+k] += temp_vectors_error[j][(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]*(float)(-(float)(sqrtf((float)var[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]+epsilon)/batch_size)-(float)((input_vectors[j*rows*cols+z*cols+k]-mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad])*2*(float)(1-(float)1/batch_size)*(input_vectors[j*rows*cols+z*cols+k]-mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad])-(float)(2/batch_size)*(temp_array[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]-input_vectors[j*rows*cols+z*cols+k]+mean[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]))/(float)((2*batch_size)*(pow((double)var[(z-rows_pad)*(cols-2*cols_pad)+k-cols_pad]+epsilon,3/2))));
                }
            }
        }
    }

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

    for(i = 0, j = 0; j < n_bns; i+=stride,j++){
        channel_normalization_feed_forward(n_channels,&tensor[i],bns[j]->temp_vectors, bns[j]->vector_dim, bns[j]->gamma, bns[j]->beta, bns[j]->mean, bns[j]->var, &post_normalization[i],bns[j]->epsilon,pad_i,pad_j,tensor_i,tensor_j);
    }
    
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
    
    for(i = 0, j = 0; j < n_bns; i+=stride,j++){
        channel_normalization_back_prop(n_channels,&tensor[i],bns[j]->temp_vectors, bns[j]->vector_dim, bns[j]->gamma, bns[j]->beta, bns[j]->mean, bns[j]->var,&ret_error[i],bns[j]->d_gamma, bns[j]->d_beta,&input_error[i], bns[j]->temp1,bns[j]->temp2, bns[j]->epsilon,pad_i,pad_j,tensor_i,tensor_j);
    }
}
