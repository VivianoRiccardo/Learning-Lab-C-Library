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
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ float* query:= the queries, dimension: dimension*k_embedding_dimension
 *             @ float* key:= the keys, dimension: dimension*k_embedding_dimension
 *             @ float* value:= the values, dimension: dimension*v_embedding_dimension
 *             @ float* score_matrix:= where we store query*key^(T)/sqrtf(k_embedding_dimension), dimension: dimension*dimension
 *             @ float* score_matrix_softmax:= for each row of the score matrix we apply the softmax, dimension: dimension*dimension
 *             @ float* output:= where we store the final output, dimension: dimension X v_embedding_dimension
 *             @ int dimension:= the dimension of q,k,v (number of tokens)
 *             @ int attention_flag:= standard attention or masked attention
 *             @ int k_embedding_dimension:= is the same of q_embedding_dimension, each token is rapresented by a vector of k_embedding_dimension for key and queries
 *             @ int v_embedding_dimension:= each token is rapresented by a vector of v_embedding_dimension for values
 * */
void self_attention_ff(float* query, float* key, float* value, float* score_matrix,float* score_matrix_softmax,float* output, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension){
    int i,j, n = dimension*k_embedding_dimension, d = dimension*dimension, m = dimension*v_embedding_dimension;
    float sqrt_dimension = sqrtf(k_embedding_dimension);
    for(i = 0; i < n; i+=k_embedding_dimension){
        for(j = 0; j < n; j+=k_embedding_dimension){
            score_matrix[i/k_embedding_dimension*dimension+j/k_embedding_dimension] = dotProduct1D(&query[i],&key[j],k_embedding_dimension)/sqrt_dimension;
        }
        if(attention_flag == STANDARD_ATTENTION)
            softmax(&score_matrix[i/k_embedding_dimension*dimension],&score_matrix_softmax[i/k_embedding_dimension*dimension],dimension);
        else if(attention_flag == MASKED_ATTENTION)
            softmax(&score_matrix[i/k_embedding_dimension*dimension],&score_matrix_softmax[i/k_embedding_dimension*dimension],i/k_embedding_dimension +1);
    }
    
    for(i = 0; i < d; i+=dimension){
        for(j = 0; j < m; j+=dimension){
            output[i/dimension*v_embedding_dimension + j/dimension] += dotProduct1D(&score_matrix_softmax[i],&value[j],dimension);
        }
    }
}

/* this function computes the "self-attention" computation explained in the attention is all you need paper for the back propagation.
 * 
 * Inputs:
 * 
 * 
 *             @ float* query:= the queries, dimension: dimension*k_embedding_dimension
 *             @ float* key:= the keys, dimension: dimension*k_embedding_dimension
 *             @ float* value:= the values, dimension: dimension*v_embedding_dimension
 *             @ float* query_error:= the queries error, dimension: dimension*k_embedding_dimension
 *             @ float* key_error:= the keys error, dimension: dimension*k_embedding_dimension
 *             @ float* value_error:= the values error, dimension: dimension*v_embedding_dimension
 *             @ float* score_matrix:= where we store query*key^(T)/sqrtf(dimension), dimension: dimension*dimension
 *             @ float* score_matrix_softmax:= for each row of the score matrix we apply the softmax, dimension: dimension*dimension
 *             @ float* score_matrix_error:= where we store query*key^(T)/sqrtf(dimension) error, dimension: dimension*dimension
 *             @ float* score_matrix_softmax_error:= for each row of the score matrix we apply the softmax (we store the error), dimension: dimension*dimension
 *             @ float* output_error:= where the output error coming from next layer, dimension: dimension*v_embedding_dimension
 *             @ int dimension:= the dimension of q,k,v (number of tokens)
 *             @ int attention_flag:= standard_attention or masked_attention
 *             @ int k_embedding_dimension:= the embedding dimension for queries and keys
 *             @ int v_embedding_dimension:= the embedding dimension for queries and values
 * */
 
void self_attention_bp(float* query, float* key, float* value, float* query_error, float* key_error, float* value_error, float* score_matrix,float* score_matrix_softmax,float* score_matrix_error,float* score_matrix_softmax_error,float* output_error, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension){
    int i,j, n = dimension*k_embedding_dimension, d = dimension*dimension, m = dimension*v_embedding_dimension;
    float sqrt_dimension = sqrtf(k_embedding_dimension);
    
    for(i = 0; i < d; i+=dimension){
        for(j = 0; j < m; j+=dimension){
            additional_mul_value(&value[j],output_error[i/dimension*v_embedding_dimension + j/dimension],&score_matrix_softmax_error[i],dimension);
            additional_mul_value(&score_matrix_softmax[i],output_error[i/dimension*v_embedding_dimension + j/dimension],&value_error[j],dimension);
        }
    }

    for(i = 0; i < n; i+=k_embedding_dimension){
        if(attention_flag == STANDARD_ATTENTION)
        derivative_softmax(&score_matrix_error[i/k_embedding_dimension*dimension],&score_matrix_softmax[i/k_embedding_dimension*dimension],&score_matrix_softmax_error[i/k_embedding_dimension*dimension],dimension);
        else if(attention_flag == MASKED_ATTENTION)
        derivative_softmax(&score_matrix_error[i/k_embedding_dimension*dimension],&score_matrix_softmax[i/k_embedding_dimension*dimension],&score_matrix_softmax_error[i/k_embedding_dimension*dimension],i/k_embedding_dimension+1);
    }
    for(i = 0; i < n; i+=k_embedding_dimension){
        for(j = 0; j < n; j+=k_embedding_dimension){
            float val = score_matrix_error[i/k_embedding_dimension*dimension+j/k_embedding_dimension]/sqrt_dimension;
            additional_mul_value(&query[i],val,&key_error[j],k_embedding_dimension);
            additional_mul_value(&key[j],val,&query_error[i],k_embedding_dimension);
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
 *             @ float* output:= the final output, dimension: n_head*dimension*v_embedding
 *             @ int dimension:= dimension of each query...
 *             @ int n_head:= number of queries....
 *             @ int output_dimension:= n_head*dimension*v_embedding
 *             @ int attention_flag:= standard attention or masked attention
 *             @ int k_embedding_dimension:= the embedding dimension for each token of queries and keys
 *             @ int v_embedding_dimension:= the embedding dimension for each token of values
 *
 * why the final copy of the array? a numpy-like tensor is not implemented yet so we can't just change the shape, and stride.
 * at the end of the attention we have 
 * 
 * embedding_token1_head1 embedding_token2_head1 ... embedding_tokenn_head1
 * embedding_token1_head2 embedding_token2_head2 ... embedding_tokenn_head2
 *                                  .
 *                                  .
 *                                  .
 * embedding_token1_headm embedding_token2_headm ... embedding_tokenn_headm
 * 
 * we reshape the tensor in 
 * 
 * embedding_token1_head1 embedding_token1_head2 ... embedding_token1_headm
 * embedding_token2_head1 embedding_token2_head2 ... embedding_token2_headm
 *                                  .
 *                                  .
 *                                  .
 * embedding_tokenn_head1 embedding_tokenn_head2 ... embedding_tokenn_headm
 * 
 * 
 * 
 * 
 * */
void multi_head_attention_ff(model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output, int dimension, int n_heads, int output_dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension){
    
    int i,j;
    for(i = 0; i < n_heads; i++){
        self_attention_ff(queries[i]->output_layer,keys[i]->output_layer,values[i]->output_layer,&score_matrices[i*dimension*dimension],&score_matrices_softmax[i*dimension*dimension],&output[i*dimension*v_embedding_dimension],dimension, attention_flag,k_embedding_dimension, v_embedding_dimension);
    }
    float* temp = (float*)calloc(n_heads*dimension*v_embedding_dimension,sizeof(float));
    
    for(i = 0; i < dimension; i++){
        for(j = 0; j < n_heads; j++){
            copy_array(&output[i*k_embedding_dimension + j*dimension*k_embedding_dimension],&temp[i*n_heads*v_embedding_dimension+j*v_embedding_dimension],v_embedding_dimension);
        }
    }
    copy_array(temp,output,dimension*n_heads*v_embedding_dimension);
    free(temp);
    
}

/* self-explenaitory, read above*/
void multi_head_attention_bp(float* queries_error, float* keys_error, float* values_error, float* score_matrices_error, float* score_matrices_softmax_error, model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output_error, int dimension, int n_heads, int output_dimension, int attention_flag,int k_embedding_dimension,int v_embedding_dimension){
    
    int i,j;
    uint64_t sumq = 0;
    uint64_t sumk = 0;
    uint64_t sumv = 0;
    
    float* temp = (float*)calloc(n_heads*dimension*v_embedding_dimension,sizeof(float));
    
    for(i = 0; i < dimension; i++){
        for(j = 0; j < n_heads; j++){
            copy_array(&output_error[i*n_heads*v_embedding_dimension+j*v_embedding_dimension],&temp[i*k_embedding_dimension + j*dimension*k_embedding_dimension],v_embedding_dimension);
        }
    }
    
    
    for(i = 0; i < n_heads; i++){
        self_attention_bp(queries[i]->output_layer,keys[i]->output_layer,values[i]->output_layer,&queries_error[sumq],&keys_error[sumk],&values_error[sumv],&score_matrices[i*dimension*dimension],&score_matrices_softmax[i*dimension*dimension],&score_matrices_error[i*dimension*dimension],&score_matrices_softmax_error[i*dimension*dimension],&temp[i*dimension*v_embedding_dimension],dimension, attention_flag,k_embedding_dimension,v_embedding_dimension);
        sumq+=queries[i]->output_dimension;
        sumk+=keys[i]->output_dimension;
        sumv+=values[i]->output_dimension;
    }
    free(temp);
    
}
