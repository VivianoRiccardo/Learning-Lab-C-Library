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

#ifndef __SCALED_L2_NORM_LAYERS_H__
#define __SCALED_L2_NORM_LAYERS_H__

scaled_l2_norm* scaled_l2_normalization_layer(int vector_dimension);
void free_scaled_l2_normalization_layer(scaled_l2_norm* l2);
void save_scaled_l2_norm(scaled_l2_norm* f, int n);
scaled_l2_norm* load_scaled_l2_norm(FILE* fr);
scaled_l2_norm* copy_scaled_l2_norm(scaled_l2_norm* f);
scaled_l2_norm* reset_scaled_l2_norm(scaled_l2_norm* f);
unsigned long long int size_of_scaled_l2_norm(scaled_l2_norm* f);
void paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy);
void slow_paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy, float tau);


#endif
