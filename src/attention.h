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

#ifndef __ATTENTION_H__
#define __ATTENTION_H__

void self_attention_ff(float* query, float* key, float* value, float* score_matrix,float* score_matrix_softmax,float* output, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension);
void self_attention_bp(float* query, float* key, float* value, float* query_error, float* key_error, float* value_error, float* score_matrix,float* score_matrix_softmax,float* score_matrix_error,float* score_matrix_softmax_error,float* output_error, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension);
void multi_head_attention_ff(model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output, int dimension, int n_heads, int output_dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension);
void multi_head_attention_bp(float* queries_error, float* keys_error, float* values_error, float* score_matrices_error, float* score_matrices_softmax_error, model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output_error, int dimension, int n_heads, int output_dimension, int attention_flag,int k_embedding_dimension,int v_embedding_dimension);
#endif
