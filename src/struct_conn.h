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


#ifndef __STRUCT_CONN_H__
#define __STRUCT_CONN_H__

#include "llab.h"


struct_conn* structure_connection(int id, model* m1, model* m2, rmodel* r1, rmodel* r2, transformer_encoder* e1, transformer_encoder* e2, transformer_decoder* d1, transformer_decoder* d2, transformer* t1, transformer* t2, scaled_l2_norm* l1, scaled_l2_norm* l2, vector_struct* v1, vector_struct* v2, vector_struct* v3, int input1_type, int input2_type, int output_type, int* input_temporal_index,int* input_encoder_indeces, int* input_decoder_indeces_left, int* input_decoder_indeces_down, int* input_transf_encoder_indeces, int* input_transf_decoder_indeces, int* rmodel_input_left, int* rmodel_input_down, int decoder_left_input, int decoder_down_input, int transf_dec_input, int transf_enc_input, int concatenate_flag, int input_size, int model_input_index, int temporal_encoding_model_size, int vector_index);
void reset_struct_conn(struct_conn* s);
void struct_connection_input_arrays(struct_conn* s);
void ff_struc_conn(struct_conn* s, int transformer_flag);
void bp_struc_conn(struct_conn* s, int transformer_flag, error_super_struct* e, error_super_struct* es);


#endif
