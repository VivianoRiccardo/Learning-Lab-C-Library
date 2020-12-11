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

/* this function computes the "self-attention" computation explained in the attention is all you need paper.
 * In the paper they explain that queries and keys must have the same dimension and the values another dimension,
 * however if we apply the dot product between query and key we obtain a matrix d x d. Indeed if 
 * dimension of query = dx1 and key dimension = dx1 query*key^(T) = dxd. Now if we have the values of dimension d_v
 * if we want to apply the dot product d_v must be equals to d so we must have 1 dimension for all the three components.
 * 
 * Inputs:
 * 
 * 
 *             @ float* query:= the queries, dimension: dimension
 *             @ float* key:= the keys, dimension: dimension
 *             @ float* value:= the values, dimension: d
 *             @ float* score_matrix:= where we store query*key^(T)/sqrtf(dimension), dimension: dimension*dimension
 *             @ float* score_matrix_softmax:= for each row of the score matrix we apply the softmax, dimension: dimension*dimension
 *             @ float* output:= where we store the final output, dimension: dimension
 *             @ int dimension:= the dimension of q,k,v
 * */
void self_attention_ff(float* query, float* key, float* value, float* score_matrix,float* score_matrix_softmax,float* output, int dimension){
    int i,j;
    float sqrt_dimension = sqrtf(dimension);
    for(i = 0; i < dimension; i++){
        for(j = 0; j < dimension; j++){
            score_matrix[i*dimension+j] = query[i]*key[j]/sqrt_dimension;
        }
        softmax(&score_matrix[i*dimension],&score_matrix_softmax[i*dimension],dimension);
    }
    
    for(i = 0; i < dimension; i++){
        for(j = 0; j < dimension; j++){
            output[i] += score_matrix_softmax[i*dimension+j]*value[j];
        }
    }
}

/* this function computes the "self-attention" computation explained in the attention is all you need paper for the back propagation.
 * In the paper they explain that queries and keys must have the same dimension and the values another dimension,
 * however if we apply the dot product between query and key we obtain a matrix d x d. Indeed if 
 * dimension of query = dx1 and key dimension = dx1 query*key^(T) = dxd. Now if we have the values of dimension d_v
 * if we want to apply the dot product d_v must be equals to d so we must have 1 dimension for all the three components.
 * 
 * Inputs:
 * 
 * 
 *             @ float* query:= the queries, dimension: dimension
 *             @ float* key:= the keys, dimension: dimension
 *             @ float* value:= the values, dimension: d
 *             @ float* query_error:= the queries error, dimension: dimension
 *             @ float* key_error:= the keys error, dimension: dimension
 *             @ float* value_error:= the values error, dimension: d
 *             @ float* score_matrix:= where we store query*key^(T)/sqrtf(dimension), dimension: dimension*dimension
 *             @ float* score_matrix_softmax:= for each row of the score matrix we apply the softmax, dimension: dimension*dimension
 *             @ float* score_matrix_error:= where we store query*key^(T)/sqrtf(dimension) error, dimension: dimension*dimension
 *             @ float* score_matrix_softmax_error:= for each row of the score matrix we apply the softmax (we store the error), dimension: dimension*dimension
 *             @ float* output_error:= where the output error coming from next layer, dimension: dimension
 *             @ int dimension:= the dimension of q,k,v
 * */
 
void self_attention_bp(float* query, float* key, float* value, float* query_error, float* key_error, float* value_error, float* score_matrix,float* score_matrix_softmax,float* score_matrix_error,float* score_matrix_softmax_error,float* output_error, int dimension){
    int i,j;
    float sqrt_dimension = sqrtf(dimension);
    for(i = 0; i < dimension; i++){
        for(j = 0; j < dimension; j++){
            value_error[j]+=output_error[i]*score_matrix_softmax[i*dimension+j];
            score_matrix_softmax_error[i*dimension+j] += output_error[i]*value[j]; 
        }
    }
    
    for(i = 0; i < dimension; i++){
        derivative_softmax(&score_matrix_error[i*dimension],&score_matrix_softmax[i*dimension],&score_matrix_softmax_error[i*dimension],dimension);
    }
    for(i = 0; i < dimension; i++){
        for(j = 0; j < dimension; j++){
            query_error[i] += score_matrix_error[i*dimension+j]*key[j]/sqrt_dimension;
            key_error[i] += score_matrix_error[i*dimension+j]*query[j]/sqrt_dimension;
        }
    }
    
}

/*With this function we compute the multi head attention formula.
 * 
 * 
 * Inputs:
 * 
 *             @ float* queries:= the queries, dimension: n_head*dimension
 *             @ float* keys:= the keys, dimension: n_head*dimension
 *             @ float* values:= the values, dimension: n_head*dimension
 *             @ float* score_matrices:= read above, dimension: n_head*dimension*dimension
 *             @ float* score_matrices_softmax:= read above, dimension: n_head*dimension*dimension
 *             @ float* output:= the final output, dimension: n_head*dimension
 *             @ int dimension:= dimension of each query...
 *             @ int n_head:= number of queries....
 *             @ int output_dimension:= n_head*dimension
 * 
 * */
void multi_head_attention_ff(float* queries, float* keys, float* values,float* score_matrices, float* score_matrices_softmax, float* output, int dimension, int n_heads, int output_dimension){
    if (dimension*n_heads != output_dimension){
        fprintf(stderr,"Error: n_heads*dimension (dimension of each value, queries, keys) must be equal to output_dimension\n");
        exit(1);
    }
    
    int i;
    for(i = 0; i < n_heads; i++){
        self_attention_ff(&queries[i*dimension],&keys[i*dimension],&values[i*dimension],&score_matrices[i*dimension],&score_matrices_softmax[i*dimension],&output[i*dimension],dimension);
    }
    
}

/* self-explenaitory, read above*/
void multi_head_attention_bp(float* queries_error, float* keys_error, float* values_error, float* score_matrices_error, float* score_matrices_softmax_error, float* queries, float* keys, float* values,float* score_matrices, float* score_matrices_softmax, float* output_error, int dimension, int n_heads, int output_dimension){
    if (dimension*n_heads != output_dimension){
        fprintf(stderr,"Error: n_heads*dimension (dimension of each value, queries, keys) must be equal to output_dimension\n");
        exit(1);
    }
    
    int i;
    for(i = 0; i < n_heads; i++){
        self_attention_bp(&queries[i*dimension],&keys[i*dimension],&values[i*dimension],&queries_error[i*dimension],&keys_error[i*dimension],&values_error[i*dimension],&score_matrices[i*dimension],&score_matrices_softmax[i*dimension],&score_matrices_error[i*dimension],&score_matrices_softmax_error[i*dimension],&output_error[i*dimension],dimension);
    }
    
}
