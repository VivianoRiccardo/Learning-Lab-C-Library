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

#ifndef __MULTI_CORE_GAN_MODEL_H__
#define __MULTI_CORE_GAN_MODEL_H__

void* gan_model_thread_ff_disc(void* _args);
void* gan_model_thread_ff_gen(void* _args);
void* gan_model_thread_bp_disc(void* _args);
void* gan_model_thread_bp_gen(void* _args);
void discriminator_tensor_input_ff_multicore(ganmodel** m,float** real_input, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads);
void generator_tensor_input_ff_multicore(ganmodel** m, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads);
void discriminator_tensor_input_bp_multicore(ganmodel** m,float** real_input, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads);
void generator_tensor_input_bp_multicore(ganmodel** m, float** noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int mini_batch_size, int threads, float** returning_error);


#endif
