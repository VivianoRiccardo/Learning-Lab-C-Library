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

#ifndef __STRUCT_CONN_HANDLER_H__
#define __STRUCT_CONN_HANDLER_H__



struct_conn_handler* struct_handler(int id, int struct_type_flag, int error_flag, int n_inputs, int n_outputs, vector_struct* input, struct_conn_handler** inputs, struct_conn_handler** outpus, vector_struct* output, float lambda, float huber1, float huber2, float* alpha, model* m, rmodel* r, transformer_encoder* e, vector_struct* v, scaled_l2_norm* l2);
void free_struct_handler(struct_conn_handler* s);
int there_are_no_cycles(struct_conn_handler* s, int depth);

#endif
