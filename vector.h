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

#ifndef __VECTOR_H__
#define __VECTOR_H__

vector_struct* create_vector(float* v, int v_size, int output_size, int action, int activation_flag, int dropout_flag, int index, float dropout_threshold, int input_size);
void free_vector(vector_struct* v);
void reset_vector(vector_struct* v);
vector_struct* copy_vector(vector_struct* v);
void save_vector(vector_struct* v, int n);
vector_struct* load_vector(FILE* fr);
void ff_vector(float* input1,float* input2, vector_struct* v);
float* bp_vector(float* input1,float* input2, vector_struct* v, float* output_error);
void paste_vector(vector_struct* v, vector_struct* copy);
uint64_t size_of_vector(vector_struct* v);

#endif
