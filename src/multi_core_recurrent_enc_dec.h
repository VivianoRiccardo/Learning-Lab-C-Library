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

#ifndef __MULTI_CORE_RECURRENT_ENC_DEC_H__
#define __MULTI_CORE_RECURRENT_ENC_DEC_H__

void* recurrent_enc_dec_thread_ff(void* _args);
void* recurrent_enc_dec_thread_bp(void* _args);
void ff_recurrent_enc_dec_multicore(float*** hidden_states, float*** cell_states, float*** input_model1, float*** input2_model, recurrent_enc_dec** m, int mini_batch_size, int threads);
void bp_recurrent_enc_dec_multicore(float*** hidden_states, float*** cell_states, float*** input_model1,float*** input_model2, recurrent_enc_dec** m, float*** error_model, int mini_batch_size, int threads, float**** returning_error, float*** returning_input_error1,float*** returning_input_error2);

#endif
