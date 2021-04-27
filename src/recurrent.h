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

#ifndef __RECURRENT_H__
#define __RECURRENT_H__

void lstm_ff(float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int size);
float** lstm_bp(int flag, int size, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus);
void lstm_ff_edge_popup(int** w_active_weights, int** u_active_weights, int** w_indices,int** u_indices, float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int size, float k_percentage);
float** lstm_bp_edge_popup(int flag, int size, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus, int** w_active_output_neurons, int** u_active_output_neurons, int** w_indices, int** u_indices, float** d_w_scores, float** d_u_scores, float k_percentage, int** w_active_output_neurons_up, int** u_active_output_neurons_up);

#endif
