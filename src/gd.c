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

#include "llab.h"

/* This function update a parameter p using the nesterov momentum
 * 
 * Input:
 *                @ float* p:= the parameter p that must be updated
 *                @ float lr:= the learning rate
 *                @ float m:= the momentum
 *                @ int mini_batch_size:= the size of the mini batch for sgd
 *                @ float dp:= sum of the deritavie of p over the whole mini batch
 *                @ float* delta:= the delta parameter of momentum
 * */
void nesterov_momentum(float* p, float lr, float m, int mini_batch_size, float dp, float* delta){
     float temp = (*delta);
     (*delta) = m*(*delta)-lr*(float)(dp/mini_batch_size);
     (*p) += m*m*temp - (1+m)*lr*(float)(dp/mini_batch_size);
}


/* This function update a parameter p using the adam optimization algorithm
 * 
 * Input:
 *                @ float* p:= the parameter p that must be updated
 *                @ float* delta1:= the parameter m of the adam algorithm
 *                @ float* delta2:= the parameter v of the adam algorithm
 *                @ float dp:= the sum of the derivative of p over the whole mini batch
 *                @ float lr:= the learning rate
 *                @ float b1:= hyper parameter usually 0.9
 *                @ float b2:= the hyper parameter usually 0.999
 *                @ float bb1:= b1^t where t is the time that p has been updated
 *                @ float bb2:= b2^t where t is the time that p has been updated
 *                @ float epsilon:= hyper parameter 10^-8
 *                @ int mini_batch_size:= the size of the mini batch
 * */
void adam_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size){
     float temp = (float)dp/mini_batch_size;
     (*delta1) = b1*(*delta1)+(1-b1)*temp;
     (*delta2) = b2*(*delta2) + (1-b2)*(temp*temp);
     (*p) -= ((lr*(*delta1)/(1-bb1))/(sqrtf((*delta2)/(1-bb2))+epsilon));
}


/* This function update a parameter p using the adam optimization algorithm
 * 
 * Input:
 *                @ float* p:= the parameter p that must be updated
 *                @ float* delta1:= the parameter m of the adam algorithm
 *                @ float* delta2:= the parameter v of the adam algorithm
 *                @ float dp:= the sum of the derivative of p over the whole mini batch
 *                @ float lr:= the learning rate
 *                @ float b1:= hyper parameter usually 0.9
 *                @ float b2:= the hyper parameter usually 0.999
 *                @ float bb1:= b1^t where t is the time that p has been updated
 *                @ float bb2:= b2^t where t is the time that p has been updated
 *                @ float epsilon:= hyper parameter 10^-8
 *                @ int mini_batch_size:= the size of the mini batch
 *                   @ unsigned long long int t:= the number of time radam has been used
 * */
void radam_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size, unsigned long long int t){
     float temp = (float)dp/mini_batch_size;
     float p_inf = 2/(1-b2)-1;
     (*delta1) = b1*(*delta1)+(1-b1)*temp;
     (*delta2) = b2*(*delta2) + (1-b2)*(temp*temp);
     float m_t_hat = (*delta1)/(1-bb1);
     long double p_t = p_inf-(long double)2*t*bb2/(1-bb2);
     if(p_t > RADAM_THRESHOLD){
         float v_t_hat = sqrtf((*delta2)/(1-bb2));
         float r_t = sqrtf(((p_t-4)*(p_t-2)*p_inf)/((p_inf-4)*(p_inf-2)*p_t));
         (*p) -= lr*r_t*m_t_hat/v_t_hat;
     }
     
     else 
        (*p) -= lr*m_t_hat;
}

