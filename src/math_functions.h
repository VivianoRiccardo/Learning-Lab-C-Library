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

#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__


void softmax(float* input, float* output, int size);
void derivative_softmax_array(float* input, float* output,float* softmax_arr,float* error, int size);
float sigmoid(float x);
void sigmoid_array(float* input, float* output, int size);
float derivative_sigmoid(float x);
void derivative_sigmoid_array(float* input, float* output, int size);
float relu(float x);
void relu_array(float* input, float* output, int size);
float derivative_relu(float x);
void derivative_relu_array(float* input, float* output, int size);
float leaky_relu(float x);
void leaky_relu_array(float* input, float* output, int size);
float derivative_leaky_relu(float x);
void derivative_leaky_relu_array(float* input, float* output, int size);
float tanhh(float x);
void tanhh_array(float* input, float* output, int size);
float derivative_tanhh(float x);
void derivative_tanhh_array(float* input, float* output, int size);
float mse(float y_hat, float y);
float derivative_mse(float y_hat, float y);
float cross_entropy(float y_hat, float y);
float derivative_cross_entropy(float y_hat, float y);
float cross_entropy_reduced_form(float y_hat, float y);
float derivative_cross_entropy_reduced_form_with_softmax(float y_hat, float y);
void derivative_cross_entropy_reduced_form_with_softmax_array(float* y_hat, float* y,float* output, int size);
float huber_loss(float y_hat, float y, float threshold);
float derivative_huber_loss(float y_hat, float y, float threshold);
void derivative_huber_loss_array(float* y_hat, float* y,float* output, float threshold, int size);
float modified_huber_loss(float y_hat, float y, float threshold1, float threshold2);
float derivative_modified_huber_loss(float y_hat, float y, float threshold1, float threshold2);
void derivative_modified_huber_loss_array(float* y_hat, float* y, float threshold1, float* output, float threshold2, int size);
float focal_loss(float y_hat, float y, float gamma);
void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size);
float derivative_focal_loss(float y_hat, float y, float gamma);
void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size);
void mse_array(float* y_hat, float* y, float* output, int size);
void derivative_mse_array(float* y_hat, float* y, float* output, int size);
void cross_entropy_array(float* y_hat, float* y, float* output, int size);
void derivative_cross_entropy_array(float* y_hat, float* y, float* output, int size);
void kl_divergence(float* input1, float* input2, float* output, int size);
void derivative_kl_divergence(float* y_hat, float* y, float* output, int size);
float entropy(float y_hat);
void entropy_array(float* y_hat, float* output, int size);
float derivative_entropy(float y_hat);
void derivative_entropy_array(float* y_hat, float* output, int size);

#endif
