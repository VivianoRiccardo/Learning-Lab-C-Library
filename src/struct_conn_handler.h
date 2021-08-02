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

#ifndef __STRUCT_CONN_HANDLER_H__
#define __STRUCT_CONN_HANDLER_H__

struct_conn_handler* init_mother_of_all_structs(int n_inputs, int n_models,int n_rmodels,int n_encoders,int n_decoders,int n_transformers,int n_l2s,int n_vectors,int n_total_structures,int n_struct_conn,int n_targets, model** m, rmodel** r, transformer_encoder** e,transformer_decoder** d,transformer** t,scaled_l2_norm** l2,vector_struct** v,struct_conn** s, int** models, int** rmodels,int** encoders,int** decoders,int** transformers,int** l2s,int** vectors,float** targets,int* targets_index,int* targets_error_flag,float** targets_weights,float* targets_threshold1,float* targets_threshold2,float* targets_gamma, int* targets_size);
void free_struct_conn_handler(struct_conn_handler* s);
void free_struct_conn_handler_without_learning_parameters(struct_conn_handler* s);
struct_conn_handler* copy_struct_conn_handler(struct_conn_handler* s);
struct_conn_handler* copy_struct_conn_handler_without_learning_parameters(struct_conn_handler* s);
void paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy);
void paste_struct_conn_handler_without_learning_parameters(struct_conn_handler* s, struct_conn_handler* copy);
void slow_paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy, float tau);
void reset_struct_conn_handler(struct_conn_handler* s);
void reset_struct_conn_handler_without_learning_parameters(struct_conn_handler* s);
uint64_t size_of_struct_conn_handler(struct_conn_handler* s);
uint64_t size_of_struct_conn_handler_without_learning_parameters(struct_conn_handler* s);


#endif
