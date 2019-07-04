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

/* This function, given a threshold, clips the gradient of the weights of the rmodel if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ rmodel* m:= the recurrent model
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient_rmodel(rmodel* m, float threshold) {
     float sum = 0;
     sum += sum_all_quadratic_derivative_weights_lstms(m->lstms,m->layers);
     
     sum = sqrtf(sum);
     if(sum >= threshold)
         clip_lstms(m->lstms,m->layers,threshold,sum);
     
}

/* This function, given a threshold, clips the gradient of the weights of the bmodel if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ bmodel* m:= the recurrent model
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient_bmodel(bmodel* m, float threshold) {
     float sum = 0;
     sum += sum_all_quadratic_derivative_weights_fcls(m->fcls, m->n_fcl);
     sum += sum_all_quadratic_derivative_weights_cls(m->cls, m->n_cl);
     sum += sum_all_quadratic_derivative_weights_rls(m->rls, m->n_rl);
     sum += sum_all_quadratic_derivative_weights_bns(m->bns, m->n_bn);
     
     sum = sqrtf(sum);
     if(sum >= threshold){
         clip_fcls(m->fcls, m->n_fcl, threshold, sum);
         clip_cls(m->cls, m->n_cl, threshold, sum);
         clip_rls(m->rls, m->n_rl, threshold, sum);
         clip_bns(m->bns, m->n_bn, threshold, sum);
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
        
        if(cls[j]->normalization_flag == GROUP_NORMALIZATION){
            for(k = 0; k < cls[j]->n_kernels/cls[j]->group_norm_channels; k++){
                for(z = 0; z < cls[j]->group_norm[k]->vector_dim; z++){
                    cls[j]->group_norm[k]->d_gamma[z]*=(threshold)/(norm);
                }
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

/* This functions clips the derivative weights according to the clipping_gradient formula
  * of lstm layers
  * 
  * Input:
  * 
  *             @ lstm* lstms:= lstm layers
  *             @ int n:= the number of lstm layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_lstms(lstm** lstms, int n, float threshold, float norm){
    int i,j;
    for(i = 0; i < n; i++){
        for(j = 0; j < lstms[i]->size*lstms[i]->size; j++){
            lstms[i]->d_w[0][j]*=(threshold)/(norm);
            lstms[i]->d_u[0][j]*=(threshold)/(norm);
            lstms[i]->d_w[1][j]*=(threshold)/(norm);
            lstms[i]->d_u[1][j]*=(threshold)/(norm);
            lstms[i]->d_w[2][j]*=(threshold)/(norm);
            lstms[i]->d_u[2][j]*=(threshold)/(norm);
            lstms[i]->d_w[3][j]*=(threshold)/(norm);
            lstms[i]->d_u[3][j]*=(threshold)/(norm);
        }
    }
    
}

/* This functions clips the derivative weights according to the clipping_gradient formula
  * of batch-normalized layers
  * 
  * Input:
  * 
  *             @ bn** bns:= batch normalized layers
  *             @ int n:= the number of batch normalized layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_bns(bn** bns, int n, float threshold, float norm){
    int i,j;
    for(i = 0; i < n; i++){
        for(j = 0; j < bns[i]->vector_dim; j++){
            bns[i]->d_gamma[j]*=(threshold)/(norm);
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
        
        if(cls[j]->normalization_flag == GROUP_NORMALIZATION){
            for(k = 0; k < cls[j]->n_kernels/cls[j]->group_norm_channels; k++){
                for(z = 0; z < cls[j]->group_norm[k]->vector_dim; z++){
                    temp = cls[j]->group_norm[k]->d_gamma[z];
                    sum += temp*temp;
                }
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

/* This functions returns the derivative of the weights of the lstm layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ lstms** lstms:= lstm layers
  *             @ int n:= the number of fully-connected layers
  * 
  * */
float sum_all_quadratic_derivative_weights_lstms(lstm** lstms, int n){
    int i,j;
    float sum = 0,temp;
    for(i = 0; i < n; i++){
        for(j = 0; j < lstms[i]->size*lstms[i]->size; j++){
            temp = lstms[i]->d_w[0][j];
            sum += temp*temp;
            temp = lstms[i]->d_u[0][j];
            sum += temp*temp;
            temp = lstms[i]->d_w[1][j];
            sum += temp*temp;
            temp = lstms[i]->d_u[1][j];
            sum += temp*temp;
            temp = lstms[i]->d_w[2][j];
            sum += temp*temp;
            temp = lstms[i]->d_u[2][j];
            sum += temp*temp;
            temp = lstms[i]->d_w[3][j];
            sum += temp*temp;
            temp = lstms[i]->d_u[3][j];
            sum += temp*temp;
        }
    }
    
    return sum;
}

/* This functions returns the derivative of the weights of the batch_normalized layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ bn** bns:= batch_normalized layers
  *             @ int n:= the number of batch normalized layers
  * 
  * */
float sum_all_quadratic_derivative_weights_bns(bn** bns, int n){
    int i,j;
    float sum = 0,temp;
    for(i = 0; i < n; i++){
        for(j = 0; j < bns[i]->vector_dim; j++){
            temp = bns[i]->d_gamma[j];
            sum += temp*temp;
        }
    }
    
    return sum;
}
