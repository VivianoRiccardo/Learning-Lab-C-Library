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

/* This function, given a threshold, clips the gradient of the weights of the whole model given by all the models m and rmodels r as well as the transformers given, if the ||DL/Dw|| > threshold,
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * (sub normalization layers are still considered during the clipping)
 * Input:
 * 
 *             @ model** m:= the models
 *             @ model** r:= the rmodels
 *                @ transformer** t:= the transformers
 *             @ int n_m:= the number of models
 *             @ int n_r:= the number of rmodels
 *             @ int n_t:= the number of transformers
 *             @ float threshold:= the threshold
 * 
 * */
void general_clipping_gradient(model** m, rmodel** r,transformer** t, int n_m, int n_r, int n_t, float threshold){
    double sum = 0;
    int i,j;
    for(i = 0; i < n_m; i++){
        sum += (double)sum_all_quadratic_derivative_weights_fcls(m[i]->fcls,m[i]->n_fcl);
        sum += (double)sum_all_quadratic_derivative_weights_cls(m[i]->cls,m[i]->n_cl);
        sum += (double)sum_all_quadratic_derivative_weights_rls(m[i]->rls,m[i]->n_rl);
    }
    
    for(i = 0; i < n_r; i++){
        sum += (double)sum_all_quadratic_derivative_weights_lstms(r[i]->lstms,r[i]->layers);
    }
    
    for(i = 0; i < n_t; i++){
        for(j = 0; j < t[i]->n_te; j++){
             sum += (double)sum_all_quadratic_derivative_weights_m(t[i]->te[j]->m);
             sum += (double)sum_all_quadratic_derivative_weights_m(t[i]->te[j]->linear_after_attention);
             sum += (double)sum_all_quadratic_derivative_weights_fcls(t[i]->te[j]->fcls,t[i]->te[j]->n_head*3);
             sum += (double)sum_all_quadratic_derivative_weights_scaled_l2_norm(t[i]->te[j]->l2,t[i]->te[j]->n_l2);
         }
         for(j = 0; j < t[i]->n_td; j++){
             sum += (double)sum_all_quadratic_derivative_weights_m(t[i]->td[j]->e->m);
             sum += (double)sum_all_quadratic_derivative_weights_m(t[i]->td[j]->e->linear_after_attention);
             sum += (double)sum_all_quadratic_derivative_weights_m(t[i]->td[j]->linear_after_attention);
             sum += (double)sum_all_quadratic_derivative_weights_fcls(t[i]->td[j]->e->fcls,t[i]->td[j]->e->n_head*3);
             sum += (double)sum_all_quadratic_derivative_weights_fcls(t[i]->td[j]->fcls,t[i]->td[j]->n_head*3);
             sum += (double)sum_all_quadratic_derivative_weights_scaled_l2_norm(t[i]->td[j]->e->l2,t[i]->td[j]->e->n_l2);
             sum += (double)sum_all_quadratic_derivative_weights_scaled_l2_norm(t[i]->td[j]->l2,t[i]->td[j]->n_l2);
         }
    }
    sum = sqrtl(sum);
    
    if(sum >= threshold){
        for(i = 0; i < n_m; i++){
            clip_fcls(m[i]->fcls,m[i]->n_fcl,threshold,sum);
            clip_cls(m[i]->cls,m[i]->n_cl,threshold,sum);
            clip_rls(m[i]->rls,m[i]->n_rl,threshold,sum);
        }
    }
    for(i = 0; i < n_r; i++){
        clip_lstms(r[i]->lstms,r[i]->layers,threshold,sum);
    }
    
    for(i = 0; i < n_t; i++){
        for(j = 0; j < t[i]->n_te; j++){
             clip_fcls(t[i]->te[j]->fcls,t[i]->te[j]->n_head*3,threshold,sum);
             clip_fcls(t[i]->te[j]->m->fcls,t[i]->te[j]->m->n_fcl,threshold,sum);
             clip_cls(t[i]->te[j]->m->cls,t[i]->te[j]->m->n_cl,threshold,sum);
             clip_rls(t[i]->te[j]->m->rls,t[i]->te[j]->m->n_rl,threshold,sum);
             clip_fcls(t[i]->te[j]->linear_after_attention->fcls,t[i]->te[j]->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t[i]->te[j]->linear_after_attention->cls,t[i]->te[j]->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t[i]->te[j]->linear_after_attention->rls,t[i]->te[j]->linear_after_attention->n_rl,threshold,sum);
             clip_scaled_l2(t[i]->te[j]->l2,t[i]->te[j]->n_l2,threshold,sum);
         }
         for(j = 0; j < t[i]->n_td; j++){
             clip_fcls(t[i]->td[j]->fcls,t[i]->td[j]->n_head*3,threshold,sum);
             clip_fcls(t[i]->td[j]->e->fcls,t[i]->td[j]->e->n_head*3,threshold,sum);
             clip_fcls(t[i]->td[j]->e->m->fcls,t[i]->td[j]->e->m->n_fcl,threshold,sum);
             clip_cls(t[i]->td[j]->e->m->cls,t[i]->td[j]->e->m->n_cl,threshold,sum);
             clip_rls(t[i]->td[j]->e->m->rls,t[i]->td[j]->e->m->n_rl,threshold,sum);
             clip_fcls(t[i]->td[j]->e->linear_after_attention->fcls,t[i]->td[j]->e->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t[i]->td[j]->e->linear_after_attention->cls,t[i]->td[j]->e->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t[i]->td[j]->e->linear_after_attention->rls,t[i]->td[j]->e->linear_after_attention->n_rl,threshold,sum);
             clip_fcls(t[i]->td[j]->linear_after_attention->fcls,t[i]->td[j]->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t[i]->td[j]->linear_after_attention->cls,t[i]->td[j]->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t[i]->td[j]->linear_after_attention->rls,t[i]->td[j]->linear_after_attention->n_rl,threshold,sum);
             clip_scaled_l2(t[i]->td[j]->e->l2,t[i]->td[j]->e->n_l2,threshold,sum);
             clip_scaled_l2(t[i]->td[j]->l2,t[i]->td[j]->n_l2,threshold,sum);
         }
     }
    
}
/* This function, given a threshold, clips the gradient of the weights of the model if the ||DL/Dw|| > threshold,
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

/* This function, given a threshold, sum all the quadratic derivative weights of the model m
 * 
 * Input:
 * 
 *             @ model* m:= the model
 * 
 * */
 
float sum_all_quadratic_derivative_weights_m(model* m) {
     float sum = 0;
     sum += sum_all_quadratic_derivative_weights_fcls(m->fcls, m->n_fcl);
     sum += sum_all_quadratic_derivative_weights_cls(m->cls, m->n_cl);
     sum += sum_all_quadratic_derivative_weights_rls(m->rls, m->n_rl);
     return sum;
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
     int i,j;
     sum += sum_all_quadratic_derivative_weights_lstms(m->lstms,m->layers);
     sum = sqrtf(sum);
     if(sum >= threshold)
         clip_lstms(m->lstms,m->layers,threshold,sum);   
}


/* This function, given a threshold, clips the gradient of the weights of the encoder transformer if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ transformer_encoder* t:= the encoder architecture
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient_transf_encoder(transformer_encoder* t, float threshold) {
     float sum = 0;
     int i,j;
     sum += sum_all_quadratic_derivative_weights_m(t->m);
     sum += sum_all_quadratic_derivative_weights_m(t->linear_after_attention);
     sum += sum_all_quadratic_derivative_weights_fcls(t->fcls,t->n_head*3);
     sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->l2,t->n_l2);
     sum = sqrtf(sum);
     if(sum >= threshold){
         clip_fcls(t->fcls,t->n_head*3,threshold,sum);
         clip_fcls(t->m->fcls,t->m->n_fcl,threshold,sum);
         clip_cls(t->m->cls,t->m->n_cl,threshold,sum);
         clip_rls(t->m->rls,t->m->n_rl,threshold,sum);
         clip_fcls(t->linear_after_attention->fcls,t->linear_after_attention->n_fcl,threshold,sum);
         clip_cls(t->linear_after_attention->cls,t->linear_after_attention->n_cl,threshold,sum);
         clip_rls(t->linear_after_attention->rls,t->linear_after_attention->n_rl,threshold,sum);
         clip_scaled_l2(t->l2,t->n_l2,threshold,sum);
     }   
}

/* This function, given a threshold, clips the gradient of the weights of the decoder transformer if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ transformer_decoder* t:= the decoder architecture
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient_transf_decoder(transformer_decoder* t, float threshold) {
     float sum = 0;
     int i,j;
     sum += sum_all_quadratic_derivative_weights_m(t->e->m);
     sum += sum_all_quadratic_derivative_weights_m(t->e->linear_after_attention);
     sum += sum_all_quadratic_derivative_weights_m(t->linear_after_attention);
     sum += sum_all_quadratic_derivative_weights_fcls(t->e->fcls,t->e->n_head*3);
     sum += sum_all_quadratic_derivative_weights_fcls(t->fcls,t->n_head*3);
     sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->e->l2,t->e->n_l2);
     sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->l2,t->n_l2);
     sum = sqrtf(sum);
     if(sum >= threshold){
         clip_fcls(t->fcls,t->n_head*3,threshold,sum);
         clip_fcls(t->e->fcls,t->e->n_head*3,threshold,sum);
         clip_fcls(t->e->m->fcls,t->e->m->n_fcl,threshold,sum);
         clip_cls(t->e->m->cls,t->e->m->n_cl,threshold,sum);
         clip_rls(t->e->m->rls,t->e->m->n_rl,threshold,sum);
         clip_fcls(t->e->linear_after_attention->fcls,t->e->linear_after_attention->n_fcl,threshold,sum);
         clip_cls(t->e->linear_after_attention->cls,t->e->linear_after_attention->n_cl,threshold,sum);
         clip_rls(t->e->linear_after_attention->rls,t->e->linear_after_attention->n_rl,threshold,sum);
         clip_fcls(t->linear_after_attention->fcls,t->linear_after_attention->n_fcl,threshold,sum);
         clip_cls(t->linear_after_attention->cls,t->linear_after_attention->n_cl,threshold,sum);
         clip_rls(t->linear_after_attention->rls,t->linear_after_attention->n_rl,threshold,sum);
         clip_scaled_l2(t->e->l2,t->e->n_l2,threshold,sum);
         clip_scaled_l2(t->l2,t->n_l2,threshold,sum);
     }   
}

/* This function, given a threshold, clips the gradient of the weights of the transformer if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ transformer* t:= the transformer architecture
 *             @ float threshold:= the threshold
 * 
 * */
 
void clipping_gradient_transf(transformer* t, float threshold) {
     float sum = 0;
     int i,j;
     for(i = 0; i < t->n_te; i++){
         sum += sum_all_quadratic_derivative_weights_m(t->te[i]->m);
         sum += sum_all_quadratic_derivative_weights_m(t->te[i]->linear_after_attention);
         sum += sum_all_quadratic_derivative_weights_fcls(t->te[i]->fcls,t->te[i]->n_head*3);
         sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->te[i]->l2,t->te[i]->n_l2);
     }
     for(i = 0; i < t->n_td; i++){
         sum += sum_all_quadratic_derivative_weights_m(t->td[i]->e->m);
         sum += sum_all_quadratic_derivative_weights_m(t->td[i]->e->linear_after_attention);
         sum += sum_all_quadratic_derivative_weights_m(t->td[i]->linear_after_attention);
         sum += sum_all_quadratic_derivative_weights_fcls(t->td[i]->e->fcls,t->td[i]->e->n_head*3);
         sum += sum_all_quadratic_derivative_weights_fcls(t->td[i]->fcls,t->td[i]->n_head*3);
         sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->td[i]->e->l2,t->td[i]->e->n_l2);
         sum += sum_all_quadratic_derivative_weights_scaled_l2_norm(t->td[i]->l2,t->td[i]->n_l2);
     }
     sum = sqrtf(sum);
     if(sum >= threshold){
         for(i = 0; i < t->n_te; i++){
             clip_fcls(t->te[i]->fcls,t->te[i]->n_head*3,threshold,sum);
             clip_fcls(t->te[i]->m->fcls,t->te[i]->m->n_fcl,threshold,sum);
             clip_cls(t->te[i]->m->cls,t->te[i]->m->n_cl,threshold,sum);
             clip_rls(t->te[i]->m->rls,t->te[i]->m->n_rl,threshold,sum);
             clip_fcls(t->te[i]->linear_after_attention->fcls,t->te[i]->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t->te[i]->linear_after_attention->cls,t->te[i]->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t->te[i]->linear_after_attention->rls,t->te[i]->linear_after_attention->n_rl,threshold,sum);
             clip_scaled_l2(t->te[i]->l2,t->te[i]->n_l2,threshold,sum);
         }
         for(i = 0; i < t->n_td; i++){
             clip_fcls(t->td[i]->fcls,t->td[i]->n_head*3,threshold,sum);
             clip_fcls(t->td[i]->e->fcls,t->td[i]->e->n_head*3,threshold,sum);
             clip_fcls(t->td[i]->e->m->fcls,t->td[i]->e->m->n_fcl,threshold,sum);
             clip_cls(t->td[i]->e->m->cls,t->td[i]->e->m->n_cl,threshold,sum);
             clip_rls(t->td[i]->e->m->rls,t->td[i]->e->m->n_rl,threshold,sum);
             clip_fcls(t->td[i]->e->linear_after_attention->fcls,t->td[i]->e->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t->td[i]->e->linear_after_attention->cls,t->td[i]->e->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t->td[i]->e->linear_after_attention->rls,t->td[i]->e->linear_after_attention->n_rl,threshold,sum);
             clip_fcls(t->td[i]->linear_after_attention->fcls,t->td[i]->linear_after_attention->n_fcl,threshold,sum);
             clip_cls(t->td[i]->linear_after_attention->cls,t->td[i]->linear_after_attention->n_cl,threshold,sum);
             clip_rls(t->td[i]->linear_after_attention->rls,t->td[i]->linear_after_attention->n_rl,threshold,sum);
             clip_scaled_l2(t->td[i]->e->l2,t->td[i]->e->n_l2,threshold,sum);
             clip_scaled_l2(t->td[i]->l2,t->td[i]->n_l2,threshold,sum);
         }
     } 
}


/* This function, given a threshold, clips the gradient of the weights of the vaemodel if the ||DL/Dw|| > threshold
 * in that case DL/Dw_i *= threshold/||DL/Dw||
 * 
 * Input:
 * 
 *             @ vaemodel* vm:= the variational autoencoder model
 *             @ float threshold:= the threshold
 * 
 * */
void clipping_gradient_vae_model(vaemodel* vm, float threshold) {
     float sum = 0;
     sum += sum_all_quadratic_derivative_weights_fcls(vm->encoder->fcls, vm->encoder->n_fcl);
     sum += sum_all_quadratic_derivative_weights_cls(vm->encoder->cls, vm->encoder->n_cl);
     sum += sum_all_quadratic_derivative_weights_rls(vm->encoder->rls, vm->encoder->n_rl);
     sum += sum_all_quadratic_derivative_weights_fcls(vm->decoder->fcls, vm->decoder->n_fcl);
     sum += sum_all_quadratic_derivative_weights_cls(vm->decoder->cls, vm->decoder->n_cl);
     sum += sum_all_quadratic_derivative_weights_rls(vm->decoder->rls, vm->decoder->n_rl);
     
     sum = sqrtf(sum);
     if(sum >= threshold){
         clip_fcls(vm->encoder->fcls, vm->encoder->n_fcl, threshold, sum);
         clip_cls(vm->encoder->cls, vm->encoder->n_cl, threshold, sum);
         clip_rls(vm->encoder->rls, vm->encoder->n_rl, threshold, sum);
         clip_fcls(vm->decoder->fcls, vm->decoder->n_fcl, threshold, sum);
         clip_cls(vm->decoder->cls, vm->decoder->n_cl, threshold, sum);
         clip_rls(vm->decoder->rls, vm->decoder->n_rl, threshold, sum);
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
        if(cls[j]->convolutional_flag == CONVOLUTION){
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
        if(fcls[i]->normalization_flag == LAYER_NORMALIZATION)
            clip_bns(&fcls[i]->layer_norm,fcls[i]->output/fcls[i]->n_groups,threshold,norm);
        
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
        if(lstms[i]->norm_flag == GROUP_NORMALIZATION)
            clip_bns(lstms[i]->bns,lstms[i]->window/lstms[i]->n_grouped_cell,threshold,norm);
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

/* This functions clips the derivative weights according to the clipping_gradient formula
  * of scaled l2 norm layers
  * 
  * Input:
  * 
  *             @ bn** bns:= scaled l2 norm layers
  *             @ int n:= the number of batch normalized layers
  *             @ float threshold:= the threshold of the clipping gradient formula
  *             @ float norm:= the ||DL/Dw|| of the entire network
  * 
  * */
void clip_scaled_l2(scaled_l2_norm** l, int n, float threshold, float norm){
    int i,j;
    for(i = 0; i < n; i++){
        l[i]->d_learned_g*=(threshold)/(norm);
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
        if(cls[j]->convolutional_flag == CONVOLUTION || cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
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
    }
    return sum;
}

/* This functions returns the derivative of the weights of the scaled l2 norm layers in quadratic form
 * returns Sum for all i (DL/Dw_i^2)    
  * 
  * Input:
  * 
  *             @ scaled_l2_norm** l:= l2 layers
  *             @ int n:= the number of l2 layers
  * 
  * */
float sum_all_quadratic_derivative_weights_scaled_l2_norm(scaled_l2_norm** l, int n){
    int i;
    float sum = 0;
    for(i = 0; i < n; i++){
        sum+=l[i]->d_learned_g*l[i]->d_learned_g;
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
        if(fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < fcls[i]->output*fcls[i]->input; j++){
                temp = fcls[i]->d_weights[j];
                sum += temp*temp;
            }
        }
        if(fcls[i]->normalization_flag == LAYER_NORMALIZATION)
            sum += sum_all_quadratic_derivative_weights_bns(&fcls[i]->layer_norm,1);
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
        if(lstms[i]->norm_flag == GROUP_NORMALIZATION)
            sum += sum_all_quadratic_derivative_weights_bns(lstms[i]->bns,lstms[i]->window/lstms[i]->n_grouped_cell);
        
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


/* https://arxiv.org/abs/2102.06171 */
void adaptive_gradient_clipping_lstm(lstm* f ,float threshold, float epsilon){
    int i,j,k;
    float sum_w;
    float sum_g;
    float ratio;
    if(f->training_mode == GRADIENT_DESCENT){
        for(k = 0; k < 4; k++){
            for(i = 0, sum_w = 0, sum_g = 0; i < f->size; i++){
                for(j = 0; j < f->size; j++){
                    sum_w+=f->w[k][i*f->size+j]*f->w[k][i*f->size+j];
                    sum_g+=f->d_w[k][i*f->size+j]*f->d_w[k][i*f->size+j];
                }
                sum_w = sqrtf(sum_w);
                sum_g = sqrtf(sum_g);
                sum_w = max(sum_w,epsilon);
                ratio = sum_w/sum_g;
                if(ratio > threshold){
                    ratio*=threshold;
                    for(j = 0; j < f->size; j++){
                        f->d_w[k][i*f->size+j]*=ratio;
                    }    
                }
            }
            for(i = 0, sum_w = 0, sum_g = 0; i < f->size; i++){
                for(j = 0; j < f->size; j++){
                    sum_w+=f->u[k][i*f->size+j]*f->u[k][i*f->size+j];
                    sum_g+=f->d_u[k][i*f->size+j]*f->d_u[k][i*f->size+j];
                }
                sum_w = sqrtf(sum_w);
                sum_g = sqrtf(sum_g);
                sum_w = max(sum_w,epsilon);
                ratio = sum_w/sum_g;
                if(ratio > threshold){
                    ratio*=threshold;
                    for(j = 0; j < f->size; j++){
                        f->d_u[k][i*f->size+j]*=ratio;
                    }    
                }
            }
        }
    }
    if(f->training_mode == EDGE_POPUP){
        for(k = 0; k < 4; k++){
            for(i = 0, sum_w = 0, sum_g = 0; i < f->size; i++){
                for(j = 0; j < f->size; j++){
                    sum_w+=f->w_scores[k][i*f->size+j]*f->w_scores[k][i*f->size+j];
                    sum_g+=f->d_w_scores[k][i*f->size+j]*f->d_w_scores[k][i*f->size+j];
                }
                sum_w = sqrtf(sum_w);
                sum_g = sqrtf(sum_g);
                sum_w = max(sum_w,epsilon);
                ratio = sum_w/sum_g;
                if(ratio > threshold){
                    ratio*=threshold;
                    for(j = 0; j < f->size; j++){
                        f->d_w_scores[k][i*f->size+j]*=ratio;
                    }    
                }
            }
            for(i = 0, sum_w = 0, sum_g = 0; i < f->size; i++){
                for(j = 0; j < f->size; j++){
                    sum_w+=f->u_scores[k][i*f->size+j]*f->u_scores[k][i*f->size+j];
                    sum_g+=f->d_u_scores[k][i*f->size+j]*f->d_u_scores[k][i*f->size+j];
                }
                sum_w = sqrtf(sum_w);
                sum_g = sqrtf(sum_g);
                sum_w = max(sum_w,epsilon);
                ratio = sum_w/sum_g;
                if(ratio > threshold){
                    ratio*=threshold;
                    for(j = 0; j < f->size; j++){
                        f->d_u_scores[k][i*f->size+j]*=ratio;
                    }    
                }
            }
        }
    }
    
    
    else
        return;
}

/* https://arxiv.org/abs/2102.06171 */
void adaptive_gradient_clipping_fcl(fcl* f ,float threshold, float epsilon){
    int i,j;
    float sum_w;
    float sum_g;
    float ratio;
    if(f->training_mode == GRADIENT_DESCENT){
        for(i = 0, sum_w = 0, sum_g = 0; i < f->output; i++){
            for(j = 0; j < f->input; j++){
                sum_w+=f->weights[i*f->input+j]*f->weights[i*f->input+j];
                sum_g+=f->d_weights[i*f->input+j]*f->d_weights[i*f->input+j];
            }
            sum_w = sqrtf(sum_w);
            sum_g = sqrtf(sum_g);
            sum_w = max(sum_w,epsilon);
            ratio = sum_w/sum_g;
            if(ratio > threshold){
                ratio*=threshold;
                for(j = 0; j < f->input; j++){
                    f->d_weights[i*f->input+j]*=ratio;
                }    
            }
        }
    }
    else if(f->training_mode == EDGE_POPUP){
        for(i = 0, sum_w = 0, sum_g = 0; i < f->output; i++){
            for(j = 0; j < f->input; j++){
                sum_w+=f->scores[i*f->input+j]*f->scores[i*f->input+j];
                sum_g+=f->d_scores[i*f->input+j]*f->d_scores[i*f->input+j];
            }
            sum_w = sqrtf(sum_w);
            sum_g = sqrtf(sum_g);
            sum_w = max(sum_w,epsilon);
            ratio = sum_w/sum_g;
            if(ratio > threshold){
                ratio*=threshold;
                for(j = 0; j < f->input; j++){
                    f->d_scores[i*f->input+j]*=ratio;
                }    
            }
        }
    }
    
    else
        return;
}
/* https://arxiv.org/abs/2102.06171 */
void adaptive_gradient_clipping_cl(cl* f ,float threshold, float epsilon){
    int i,j;
    float sum_w;
    float sum_g;
    float ratio;
    if(f->training_mode == GRADIENT_DESCENT){
        for(i = 0, sum_w = 0, sum_g = 0; i < f->n_kernels; i++){
            for(j = 0; j < f->kernel_rows*f->kernel_cols*f->channels; j++){
                sum_w+=f->kernels[i][j]*f->kernels[i][j];
                sum_g+=f->d_kernels[i][j]*f->d_kernels[i][j];
            }
            sum_w = sqrtf(sum_w);
            sum_g = sqrtf(sum_g);
            sum_w = max(sum_w,epsilon);
            ratio = sum_w/sum_g;
            if(ratio > threshold){
                ratio*=threshold;
                for(j = 0; j < f->kernel_rows*f->kernel_cols*f->channels; j++){
                    f->d_kernels[i][j]*=ratio;
                }    
            }
        }
    }
    else if(f->training_mode == EDGE_POPUP){
        for(i = 0, sum_w = 0, sum_g = 0; i < f->n_kernels; i++){
            
            sum_w+=f->scores[i]*f->scores[i];
            sum_g+=f->d_scores[i]*f->d_scores[i];
            
            sum_w = sqrtf(sum_w);
            sum_g = sqrtf(sum_g);
            sum_w = max(sum_w,epsilon);
            ratio = sum_w/sum_g;
            if(ratio > threshold){
                ratio*=threshold;
                f->d_scores[i]*=ratio;
            }
        }
    }
    
    else
        return;
}
/* https://arxiv.org/abs/2102.06171 */
void adaptive_gradient_clipping_rl(rl* f ,float threshold, float epsilon){
    int i;
    for(i = 0; i < f->n_cl; i++){
        adaptive_gradient_clipping_cl(f->cls[i],threshold,epsilon);
    }
}


void adaptive_gradient_clipping_model(model* m, float threshold, float epsilon){
    int i;
    for(i = 0; i < m->n_fcl; i++){
        adaptive_gradient_clipping_fcl(m->fcls[i],threshold,epsilon);
    }
    for(i = 0; i < m->n_cl; i++){
        adaptive_gradient_clipping_cl(m->cls[i],threshold,epsilon);
    }
    for(i = 0; i < m->n_rl; i++){
        adaptive_gradient_clipping_rl(m->rls[i],threshold,epsilon);
    }
}

void adaptive_gradient_clipping_rmodel(rmodel* r, float threshold, float epsilon){
    int i;
    for(i = 0; i < r->n_lstm; i++){
        adaptive_gradient_clipping_lstm(r->lstms[i],threshold,epsilon);
    }
}


void adaptive_gradient_clipping_encoder_transformer(transformer_encoder* e, float threshold, float epsilon){
    int i;
    for(i = 0; i < e->n_head*3; i++){
        adaptive_gradient_clipping_fcl(e->fcls[i],threshold,epsilon);
    }
    adaptive_gradient_clipping_model(e->m,threshold,epsilon);
    adaptive_gradient_clipping_model(e->linear_after_attention,threshold,epsilon);
    
}


void adaptive_gradient_clipping_decoder_transformer(transformer_decoder* t, float threshold, float epsilon){
    adaptive_gradient_clipping_encoder_transformer(t->e,threshold,epsilon);
    int i;
    for(i = 0; i < t->n_head; i++){
        adaptive_gradient_clipping_fcl(t->fcls[i],threshold,epsilon);
    }
    adaptive_gradient_clipping_model(t->linear_after_attention,threshold,epsilon);
}


void adaptive_gradient_clipping_transformer(transformer* t, float threshold, float epsilon){
    int i;
    for(i = 0; i < t->n_te; i++){
        adaptive_gradient_clipping_encoder_transformer(t->te[i],threshold,epsilon);
    }
    for(i = 0; i < t->n_td; i++){
        adaptive_gradient_clipping_decoder_transformer(t->td[i],threshold,epsilon);
    }
}

