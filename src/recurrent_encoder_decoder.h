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

#ifndef __RECURRENT_ENCODER_DECODER_H__
#define __RECURRENT_ENCODER_DECODER_H__

recurrent_enc_dec* recurrent_enc_dec_network(rmodel* encoder, rmodel* decoder);
void free_recurrent_enc_dec(recurrent_enc_dec* r);
recurrent_enc_dec* copy_recurrent_enc_dec(recurrent_enc_dec* r);
void paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy);
void slow_paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy, float tau);
void reset_recurrent_enc_dec(recurrent_enc_dec* r);
void save_recurrent_enc_dec(recurrent_enc_dec* r, int n1, int n2, int n3);
recurrent_enc_dec* load_recurrent_enc_dec(char* file1, char* file2, char* file3);
void ff_decoder_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int size, int layers, lstm** lstms, recurrent_enc_dec* rec);
float*** bp_decoder_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error, recurrent_enc_dec* rec);
void paste_w_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy);
void heavy_save_recurrent_enc_dec(recurrent_enc_dec* r, int n1, int n2, int n3);
recurrent_enc_dec* heavy_load_recurrent_enc_dec(char* file1, char* file2, char* file3);
int count_weights_recurrent_enc_dec(recurrent_enc_dec* m);
void ff_recurrent_dec(float** hidden_states, float** cell_states, float** input_model, recurrent_enc_dec* rec);
float*** bp_recurrent_dec(float** hidden_states, float** cell_states, float** input_model, float** error_model, recurrent_enc_dec* rec, float** input_error);
void ff_recurrent_enc_dec(float** hidden_states, float** cell_states, float** input_model1, float** input_model2, recurrent_enc_dec* rec);
float*** bp_encoder_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error, float*** dfioc, float** dropout_mask_dec, lstm** first_dec_orizontal);
float*** bp_recurrent_enc_dec(float** hidden_states, float** cell_states, float** input_model1, float** input_model2, float** error_model, recurrent_enc_dec* rec, float** input_error1,float** input_error2);
void update_recurrent_enc_dec_model(recurrent_enc_dec* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void sum_recurrent_enc_dec_partial_derivatives(recurrent_enc_dec* rec1,recurrent_enc_dec* rec2,recurrent_enc_dec* rec3);
void sum_recurrent_enc_decs_partial_derivatives(recurrent_enc_dec* sum, recurrent_enc_dec** rec, int n_models);

#endif
