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

#ifndef __RMODEL_H__
#define __RMODEL_H__

rmodel* recurrent_network(int layers, int n_lstm, lstm** lstms, int window, int hidden_state_mode);
void free_rmodel(rmodel* m);
rmodel* copy_rmodel(rmodel* m);
void paste_rmodel(rmodel* m, rmodel* copy);
void slow_paste_rmodel(rmodel* m, rmodel* copy, float tau);
rmodel* reset_rmodel(rmodel* m);
void save_rmodel(rmodel* m, int n);
void heavy_save_rmodel(rmodel* m, int n);
rmodel* load_rmodel(char* file);
rmodel* heavy_load_rmodel(char* file);
void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int size, int layers, lstm** lstms);
float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error);
int count_weights_rmodel(rmodel* m);
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3);
float* lstm_dinput(int index, int output, float** returning_error, lstm* lstms);
float* lstm_dh(int index, int output, float** returning_error, lstm* lstms);
void ff_rmodel(float** hidden_states, float** cell_states, float** input_model, rmodel* m);
float*** bp_rmodel(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error);
void paste_w_rmodel(rmodel* m, rmodel* copy);
void sum_rmodels_partial_derivatives(rmodel* m, rmodel** m2, int n_models);

#endif
