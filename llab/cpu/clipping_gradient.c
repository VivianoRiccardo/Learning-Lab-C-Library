#include "llab.h"

/* This function, given a threshold, clips the gradient of the weights of the model if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ model* m:= the model
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient(model* m, float threshold) {
     float sum = 0;
     sum += sum_all_quadratic_derivative_weights_fcls(m->fcls, m->n_fcl);
     sum += sum_all_quadratic_derivative_weights_cls(m->cls, m->n_cl);
     sum += sum_all_quadratic_derivative_weights_rls(m->rls, m->n_rl);
     
     sum = sqrtf(sum);
     if(sum >= threshold){
         clip_fcls(m->fcls, m->n_fcl, threshold, sum);
         clip_cls(m->cls, m->n_cl, threshold, sum);
         clip_rls(m->rls, m->n_rl, threshold, sum);
     }
}
 
/* This functions clips the derivative weights according to the clipping_gradient formula
  * of residual layers
  * 
  * Input:
  * 
  *             @ rl* rls:= residual layers
  *             @ int n:= the number of residual layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_rls(rl** rls, int n, float threshold,float norm){
    int i;
    for(i = 0; i < n; i++){
        clip_cls(rls[i]->cls, rls[i]->n_cl, threshold, norm);
    }
}
 
 
/* This functions clips the derivative weights according to the clipping_gradient formula
  * of convolutional layers
  * 
  * Input:
  * 
  *             @ cl* cls:= convolutional layers
  *             @ int n:= the number of convolutional layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_cls(cl** cls, int n, float threshold, float norm){
    int j,k,z;
    for(j = 0; j < n; j++){
        for(k = 0; k < cls[j]->n_kernels; k++){
            for(z = 0; z < cls[j]->channels*cls[j]->kernel_rows*cls[j]->kernel_cols; z++){
                cls[j]->d_kernels[k][z]*=(threshold)/(norm);
            }
        }
    }
    
}


/* This functions clips the derivative weights according to the clipping_gradient formula
  * of fully-connected layers
  * 
  * Input:
  * 
  *             @ fcl* fcls:= fully-connected layers
  *             @ int n:= the number of fully-connected layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_fcls(fcl** fcls, int n, float threshold, float norm){
    int i,j;
    for(i = 0; i < n; i++){
        for(j = 0; j < fcls[i]->output*fcls[i]->input; j++){
            fcls[i]->d_weights[j]*=(threshold)/(norm);
        }
    }
    
}

/* This functions returns the derivative of the weights of the residual layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ rl** rls:= residual layers
  *             @ int n:= the number of residual layers
  * 
  * */
float sum_all_quadratic_derivative_weights_rls(rl** rls, int n){
    int i;
    float sum = 0;
    for(i = 0; i < n; i++){
        sum+= sum_all_quadratic_derivative_weights_cls(rls[i]->cls, rls[i]->n_cl);
    }
    return sum;
}

/* This functions returns the derivative of the weights of the convolutional layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ cl** cls:= convolutional layers
  *             @ int n:= the number of convolutional layers
  * 
  * */
float sum_all_quadratic_derivative_weights_cls(cl** cls, int n){
    int j,k,z;
    float sum = 0,temp;
    for(j = 0; j < n; j++){
        for(k = 0; k < cls[j]->n_kernels; k++){
            for(z = 0; z < cls[j]->channels*cls[j]->kernel_rows*cls[j]->kernel_cols; z++){
                temp = cls[j]->d_kernels[k][z];
                sum += temp*temp;
            }
        }
    }
    return sum;
}


/* This functions returns the derivative of the weights of the fully-connected layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ fcl** fcls:= fully-connected layers
  *             @ int n:= the number of fully-connected layers
  * 
  * */
float sum_all_quadratic_derivative_weights_fcls(fcl** fcls, int n){
    int i,j;
    float sum = 0,temp;
    for(i = 0; i < n; i++){
        for(j = 0; j < fcls[i]->output*fcls[i]->input; j++){
            temp = fcls[i]->d_weights[j];
            sum += temp*temp;
        }
    }
    
    return sum;
}
