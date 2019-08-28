/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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

#ifndef __GAN_MODEL_H__
#define __GAN_MODEL_H__

ganmodel* gan_network(model* generator, model* discriminator,float g_lr, float g_momentum, int mini_batch_size, int g_gradient_descent_flag, float g_b1, float g_b2, int g_regularization, float g_lambda,float d_lr, float d_momentum, int d_gradient_descent_flag, float d_b1, float d_b2, int d_regularization, float d_lambda);
void free_ganmodel(ganmodel* gm);
ganmodel* copy_ganmodel(ganmodel* gm);
void paste_ganmodel(ganmodel* m1, ganmodel* m2);
void slow_paste_ganmodel(ganmodel* m1, ganmodel* m2, float tau);
void reset_ganmodel(ganmodel* m1);
unsigned long long int size_of_ganmodel(ganmodel* m1);
void save_ganmodel(ganmodel* gm, int n, int m);
ganmodel* load_ganmodel(char* file1, char* file2);
void discriminator_feed_forward(ganmodel* gm, float* real_input, float* noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j);
void discriminator_back_propagation(ganmodel* gm, float* real_input, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j);
void generator_feed_forward(ganmodel* gm, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j);
float* generator_back_propagation(ganmodel* gm, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int generator_output_size);
void update_discriminator(ganmodel* gm);
void update_generator(ganmodel* gm);
#endif
