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

#ifndef __TRANSFORMER_ENCODER_H__
#define __TRANSFORMER_ENCODER_H__

transformer_encoder* transformer_encoder_layer(model* m, model* linear_after_attention, fcl** fcls, scaled_l2_norm** l2, int input_dimension, int n_head,int residual_flag1,int normalization_flag1,int residual_flag2,int normalization_flag2, int attention_flag);
void free_transformer_encoder_layer(transformer_encoder* t);
void free_transformer_wrapped_encoder_layer(transformer_encoder* t);
void save_transformer_encoder(transformer_encoder* t, int n);
transformer_encoder* load_transformer_encoder(FILE* fr);
transformer_encoder* copy_transformer_encoder(transformer_encoder* t);
void reset_transformer_encoder(transformer_encoder* t);
void reset_transformer_encoder_for_edge_popup(transformer_encoder* t);
unsigned long long int size_of_transformer_encoder(transformer_encoder* t);
void paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy);
void slow_paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy, float tau);
void encoder_transformer_ff(float* inputs, transformer_encoder* t, int input_dimension);
float* encoder_transformer_bp(float* inputs, transformer_encoder* t, int input_dimension,float* output_error);
void reset_transformer_encoder_except_partial_derivatives(transformer_encoder* t);

#endif
