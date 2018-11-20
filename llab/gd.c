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
     (*p) += -m*temp+(1+m)*(*delta);
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
