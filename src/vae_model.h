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

#ifndef __VAE_MODEL_H__
#define __VAE_MODEL_H__

vaemodel* variational_auto_encoder_model(model* encoder, model* decoder, int latent_size);
void free_vae_model(vaemodel* vm);
vaemodel* copy_vae_model(vaemodel* vm);
void paste_vae_model(vaemodel* vm1, vaemodel* vm2);
void slow_paste_vae_model(vaemodel* vm1, vaemodel* vm2, float tau);
void reset_vae_model(vaemodel* vm);
unsigned long long int size_of_vae_model(vaemodel* vm);
void save_vae_model(vaemodel* vm, int n, int m);
vaemodel* load_vae_model(char* file1, char* file2);
void vae_model_tensor_input_ff(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j,float* input);
float* vae_model_tensor_input_bp(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension);
int count_weights_vae_model(vaemodel* vm);
void update_vae_model(vaemodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t);
void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3);

#endif
