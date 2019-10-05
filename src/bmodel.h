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

#ifndef __BMODEL_H__
#define __BMODEL_H__

bmodel* batch_network(int layers, int n_rl, int n_cl, int n_fcl, int n_bnl, rl** rls, cl** cls, fcl** fcls, bn** bnls);
void free_bmodel(bmodel* m);
bmodel* copy_bmodel(bmodel* m);
void paste_bmodel(bmodel* m, bmodel* copy);
void slow_paste_bmodel(bmodel* m, bmodel* copy, float tau);
bmodel* reset_bmodel(bmodel* m);
unsigned long long int size_of_bmodel(bmodel* m);
void save_bmodel(bmodel* m, int n);
bmodel* load_bmodel(char* file);
int count_bmodel_weights(bmodel* m);
void update_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda);
void sum_model_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3);

#endif
